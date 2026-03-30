/*
query_template: SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;

split_template: select * from dbweaver((SELECT EventTime FROM hits WHERE (CounterID=62) AND (EventDate>='2013-07-14'::DATE AND EventDate<='2013-07-15'::DATE) AND (IsRefresh=0) AND (DontCountHits=0)));
query_example: SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;

split_query: select * from dbweaver((SELECT EventTime FROM hits WHERE (CounterID=62) AND (EventDate>='2013-07-14'::DATE AND EventDate<='2013-07-15'::DATE) AND (IsRefresh=0) AND (DontCountHits=0)));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"

#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <vector>
#include <functional>

namespace duckdb {

// ============================================================
// Bind Data
// ============================================================
struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

// ============================================================
// Group Key / Hash
// ============================================================
struct GroupKey {
	timestamp_t M; // DATE_TRUNC('minute', EventTime)

	bool operator==(const GroupKey &other) const {
		return M == other.M;
	}
};

struct GroupKeyHash {
	size_t operator()(const GroupKey &k) const {
		size_t h = 0;
		h ^= std::hash<int64_t>{}(k.M.value) + 0x9e3779b9 + (h << 6) + (h >> 2);
		return h;
	}
};

// ============================================================
// Aggregation State
// ============================================================
struct AggState {
	int64_t PageViews = 0; // COUNT(*)
};

// ============================================================
// Sorting State (ORDER BY M ASC LIMIT 10 OFFSET 1000)
// ============================================================
struct SortKeyView {
	timestamp_t M;
};

struct SortState {
	std::vector<std::pair<SortKeyView, AggState>> buffer;
	bool sorted = false;

	// Fixed for this query:
	// ORDER BY M ASC LIMIT 10 OFFSET 1000
	idx_t limit = 10;
	idx_t offset = 1000;
	idx_t top_k = 1010; // offset + limit

	inline void AddRow(const GroupKey &key, const AggState &state) {
		SortKeyView view;
		view.M = key.M;
		buffer.emplace_back(view, state);
	}

	inline void SortNow() {
		if (sorted) {
			return;
		}

		// Final order: M ASC
		auto final_less = [](const std::pair<SortKeyView, AggState> &a,
		                     const std::pair<SortKeyView, AggState> &b) {
			return a.first.M < b.first.M;
		};

		// We want top_k smallest M values. Use a heap that keeps the "worst" (largest M) on top.
		// priority_queue top is element e such that Compare(x, e) == true for all x (intuitively "largest" under Compare).
		// Using compare = (a,b) => a.M < b.M yields top = largest M.
		auto heap_cmp = [](const std::pair<SortKeyView, AggState> &a,
		                   const std::pair<SortKeyView, AggState> &b) {
			return a.first.M < b.first.M;
		};

		if (top_k > 0 && top_k < buffer.size()) {
			std::priority_queue<
			    std::pair<SortKeyView, AggState>,
			    std::vector<std::pair<SortKeyView, AggState>>,
			    decltype(heap_cmp)>
			    pq(heap_cmp);

			for (auto &item : buffer) {
				if (pq.size() < top_k) {
					pq.push(item);
				} else if (item.first.M < pq.top().first.M) {
					// New item is better (smaller M) than current worst (largest M)
					pq.pop();
					pq.push(item);
				}
			}

			std::vector<std::pair<SortKeyView, AggState>> temp_result;
			temp_result.reserve(pq.size());
			while (!pq.empty()) {
				temp_result.push_back(pq.top());
				pq.pop();
			}

			std::sort(temp_result.begin(), temp_result.end(), final_less);
			buffer = std::move(temp_result);
		} else {
			std::sort(buffer.begin(), buffer.end(), final_less);
		}

		sorted = true;
	}
};

// ============================================================
// Global / Local State
// ============================================================
struct FnGlobalState : public GlobalTableFunctionState {
	std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
	SortState sort_state;

	std::mutex lock;
	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};
	std::atomic<bool> output_emitted {false};

	idx_t MaxThreads() const override {
		return std::numeric_limits<idx_t>::max();
	}
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
	std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
	bool merged = false;
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                       GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);
	return make_uniq<FnLocalState>();
}

// ============================================================
// Bind
// ============================================================
static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	// Output schema: M, PageViews
	return_types.emplace_back(LogicalType::TIMESTAMP);
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("M");
	names.emplace_back("PageViews");

	return make_uniq<FnBindData>();
}

// ============================================================
// Execute (aggregate locally)
// Input schema from split query: SELECT EventTime ...
// ============================================================
static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// Expect input.data[0] = EventTime
	UnifiedVectorFormat event_time_uvf;
	input.data[0].ToUnifiedFormat(input.size(), event_time_uvf);
	auto *event_time_ptr = reinterpret_cast<timestamp_t *>(event_time_uvf.data);

	auto &valid_event_time = event_time_uvf.validity;
	const bool event_time_all_valid = valid_event_time.AllValid();

	if (event_time_all_valid) {
		for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
			const idx_t i_event_time = event_time_uvf.sel->get_index(row_idx);
			const timestamp_t v_event_time = event_time_ptr[i_event_time];

			// DATE_TRUNC('minute', EventTime)
			const int64_t micros = v_event_time.value;
			const int64_t minute_micros = 60LL * Interval::MICROS_PER_SEC;
			const int64_t truncated_micros = (micros / minute_micros) * minute_micros;
			const timestamp_t truncated_minute(truncated_micros);

			GroupKey key;
			key.M = truncated_minute;

			auto &state = l.agg_map[key];
			state.PageViews++;
		}
	} else {
		for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
			const idx_t i_event_time = event_time_uvf.sel->get_index(row_idx);
			if (!valid_event_time.RowIsValid(i_event_time)) {
				continue;
			}

			const timestamp_t v_event_time = event_time_ptr[i_event_time];

			// DATE_TRUNC('minute', EventTime)
			const int64_t micros = v_event_time.value;
			const int64_t minute_micros = 60LL * Interval::MICROS_PER_SEC;
			const int64_t truncated_micros = (micros / minute_micros) * minute_micros;
			const timestamp_t truncated_minute(truncated_micros);

			GroupKey key;
			key.M = truncated_minute;

			auto &state = l.agg_map[key];
			state.PageViews++;
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
// Finalize (merge local -> global, single emitter outputs sorted+offset window)
// ============================================================
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                             DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	// Merge local state exactly once
	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			for (const auto &entry : l.agg_map) {
				const GroupKey &key = entry.first;
				const AggState &state = entry.second;
				g.agg_map[key].PageViews += state.PageViews;
			}
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	// Only one thread emits final result, and only after all local states have merged.
	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	bool should_emit = false;
	if (active > 0 && merged == active) {
		bool expected = false;
		should_emit = g.output_emitted.compare_exchange_strong(expected, true, std::memory_order_relaxed);
	}

	if (!should_emit) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	// Build sort buffer + sort once
	{
		std::lock_guard<std::mutex> guard(g.lock);

		// Prevent duplicate population if finalize gets re-entered somehow after emission flag was set
		if (g.sort_state.buffer.empty()) {
			g.sort_state.buffer.reserve(g.agg_map.size());
			for (const auto &entry : g.agg_map) {
				g.sort_state.AddRow(entry.first, entry.second);
			}
		}
		g.sort_state.SortNow();

		// Apply OFFSET/LIMIT on sorted grouped result
		const idx_t total_rows = g.sort_state.buffer.size();
		const idx_t start = std::min(g.sort_state.offset, total_rows);
		const idx_t end = std::min(start + g.sort_state.limit, total_rows);
		const idx_t out_count = end - start;

		// This query expects 10 rows, so fits within a single DataChunk.
		out.SetCardinality(out_count);

		for (idx_t output_idx = 0; output_idx < out_count; ++output_idx) {
			const auto &item = g.sort_state.buffer[start + output_idx];
			const SortKeyView &key = item.first;
			const AggState &state = item.second;

			out.SetValue(0, output_idx, Value::TIMESTAMP(key.M));
			out.SetValue(1, output_idx, Value::BIGINT(state.PageViews));
		}
	}

	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
// Extension Registration
// ============================================================
static void LoadInternal(ExtensionLoader &loader) {
	TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
	f.in_out_function = FnExecute;
	f.in_out_function_final = FnFinalize;
	loader.RegisterFunction(f);
}

void DbweaverExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string DbweaverExtension::Name() {
	return "dbweaver";
}

std::string DbweaverExtension::Version() const {
	return DuckDB::LibraryVersion();
}

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) {
	duckdb::LoadInternal(loader);
}
}
