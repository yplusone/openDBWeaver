/*
query_template: SELECT CounterID, AVG(STRLEN(URL)) AS l, COUNT(*) AS c
                FROM hits
                WHERE URL <> ''
                GROUP BY CounterID
                HAVING COUNT(*) > 100000
                ORDER BY l DESC
                LIMIT 25;

split_template: select * from dbweaver((SELECT URL, CounterID FROM hits WHERE (URL!='')));
query_example: SELECT CounterID, AVG(STRLEN(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;

split_query: select * from dbweaver((SELECT URL, CounterID FROM hits WHERE (URL!='')));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"

#include <atomic>
#include <limits>
#include <mutex>
#include <absl/container/flat_hash_map.h>
#include <vector>
#include <algorithm>

namespace duckdb {

// ============================================================
//  Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

// ============================================================
//  Aggregation state per group
// ============================================================

struct AggState {
	__int128 sum_len = 0; // SUM(STRLEN(URL))
	int64_t cnt = 0;      // COUNT(*)
};

// For final sort/output
struct SortRow {
	int64_t counter_id;
	int64_t cnt;
	double avg_len;
};

struct SortRowComparator {
	bool operator()(const SortRow &a, const SortRow &b) const noexcept {
		// ORDER BY l DESC
		if (a.avg_len != b.avg_len) {
			return a.avg_len > b.avg_len;
		}
		// deterministic tie-breaker
		return a.counter_id < b.counter_id;
	}
};

struct SortState {
	std::vector<SortRow> rows;
	bool sorted = false;

	inline void SortNow() {
		if (!sorted) {
			std::sort(rows.begin(), rows.end(), SortRowComparator{});
			sorted = true;
		}
	}
};

// ============================================================
//  Global / Local state
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;
	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};

	// Use a flat array for small, dense ranges of CounterID
	static constexpr int64_t MAX_FLAT = 300000;
	std::vector<AggState> agg_flat;
	absl::flat_hash_map<int64_t, AggState> agg_map;

	SortState sort_state;

	FnGlobalState() {
		agg_flat.resize(MAX_FLAT);
	}

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	std::vector<AggState> agg_flat;
	absl::flat_hash_map<int64_t, AggState> agg_map;

	FnLocalState() {
		agg_flat.resize(FnGlobalState::MAX_FLAT);
	}
};


// ============================================================
//  Init / Bind
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);
	return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	return_types.emplace_back(LogicalType::BIGINT); // CounterID
	return_types.emplace_back(LogicalType::BIGINT); // c
	return_types.emplace_back(LogicalType::DOUBLE); // l

	names.emplace_back("CounterID");
	names.emplace_back("c");
	names.emplace_back("l");

	return make_uniq<FnBindData>();
}

// ============================================================
//  Processing loops specialized by physical type and validity
// ============================================================

template <typename T, bool ALL_VALID>
static void ExecuteInternalLoop(idx_t n, UnifiedVectorFormat &url_uvf, UnifiedVectorFormat &counter_uvf,
                            FnLocalState &l) {
	auto url_ptr = (string_t *)url_uvf.data;
	auto counter_ptr = (T *)counter_uvf.data;
	auto &valid_url = url_uvf.validity;
	auto &valid_counter = counter_uvf.validity;

	for (idx_t row_idx = 0; row_idx < n; ++row_idx) {
		const idx_t i_url = url_uvf.sel->get_index(row_idx);
		const idx_t i_ctr = counter_uvf.sel->get_index(row_idx);

		if (!ALL_VALID) {
			if (!valid_url.RowIsValid(i_url) || !valid_counter.RowIsValid(i_ctr)) {
				continue;
			}
		}

		const string_t url = url_ptr[i_url];
		const idx_t url_len = url.GetSize();
		if (url_len == 0) {
			continue;
		}

		const int64_t counter_id = (int64_t)counter_ptr[i_ctr];
		if (counter_id >= 0 && (uint64_t)counter_id < l.agg_flat.size()) {
			auto &st = l.agg_flat[counter_id];
			st.sum_len += (__int128)url_len;
			st.cnt += 1;
		} else {
			auto &st = l.agg_map[counter_id];
			st.sum_len += (__int128)url_len;
			st.cnt += 1;
		}
	}
}

template <typename T>
static void ExecuteInternal(idx_t n, UnifiedVectorFormat &url_uvf, UnifiedVectorFormat &counter_uvf,
                            FnLocalState &l) {
	if (url_uvf.validity.AllValid() && counter_uvf.validity.AllValid()) {
		ExecuteInternalLoop<T, true>(n, url_uvf, counter_uvf, l);
	} else {
		ExecuteInternalLoop<T, false>(n, url_uvf, counter_uvf, l);
	}
}

// ============================================================
//  Execute: group by CounterID, accumulate sum(strlen(url)), count
// ============================================================
static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                   DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	UnifiedVectorFormat url_uvf;
	UnifiedVectorFormat counter_uvf;

	input.data[0].ToUnifiedFormat(input.size(), url_uvf);      // URL
	input.data[1].ToUnifiedFormat(input.size(), counter_uvf);  // CounterID

	const idx_t n = input.size();

	switch (input.data[1].GetType().id()) {
	case LogicalTypeId::TINYINT:   ExecuteInternal<int8_t>(n, url_uvf, counter_uvf, l); break;
	case LogicalTypeId::SMALLINT:  ExecuteInternal<int16_t>(n, url_uvf, counter_uvf, l); break;
	case LogicalTypeId::INTEGER:   ExecuteInternal<int32_t>(n, url_uvf, counter_uvf, l); break;
	case LogicalTypeId::BIGINT:    ExecuteInternal<int64_t>(n, url_uvf, counter_uvf, l); break;
	case LogicalTypeId::UTINYINT:  ExecuteInternal<uint8_t>(n, url_uvf, counter_uvf, l); break;
	case LogicalTypeId::USMALLINT: ExecuteInternal<uint16_t>(n, url_uvf, counter_uvf, l); break;
	case LogicalTypeId::UINTEGER:  ExecuteInternal<uint32_t>(n, url_uvf, counter_uvf, l); break;
	case LogicalTypeId::UBIGINT:   ExecuteInternal<uint64_t>(n, url_uvf, counter_uvf, l); break;
	default:
		throw InvalidInputException("Unsupported CounterID type: %s", input.data[1].GetType().ToString());
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize: merge local structures; last finisher sorts + emits top-25
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	// Merge local results into global state exactly once per local state
	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			for (size_t i = 0; i < l.agg_flat.size(); ++i) {
				if (l.agg_flat[i].cnt > 0) {
					g.agg_flat[i].sum_len += l.agg_flat[i].sum_len;
					g.agg_flat[i].cnt += l.agg_flat[i].cnt;
				}
			}
			for (auto &kv : l.agg_map) {
				auto &dst = g.agg_map[kv.first];
				dst.sum_len += kv.second.sum_len;
				dst.cnt += kv.second.cnt;
			}
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_release);
	}

	const auto merged = g.merged_local_states.load(std::memory_order_acquire);
	const auto active = g.active_local_states.load(std::memory_order_acquire);

	if (active > 0 && merged == active) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			if (!g.sort_state.sorted) {
				g.sort_state.rows.clear();

				// Collect from flat array: HAVING COUNT(*) > 100000
				for (size_t i = 0; i < g.agg_flat.size(); ++i) {
					if (g.agg_flat[i].cnt > 100000) {
						const double avg_len = (double)g.agg_flat[i].sum_len / (double)g.agg_flat[i].cnt;
						g.sort_state.rows.push_back(SortRow{(int64_t)i, g.agg_flat[i].cnt, avg_len});
					}
				}

				// Collect from map: HAVING COUNT(*) > 100000
				for (auto const &kv : g.agg_map) {
					if (kv.second.cnt > 100000) {
						const double avg_len = (double)kv.second.sum_len / (double)kv.second.cnt;
						g.sort_state.rows.push_back(SortRow{kv.first, kv.second.cnt, avg_len});
					}
				}

				g.sort_state.SortNow();
			}
		}

		const idx_t out_n = MinValue<idx_t>((idx_t)g.sort_state.rows.size(), 25);
		out.SetCardinality(out_n);

		for (idx_t i = 0; i < out_n; i++) {
			const auto &r = g.sort_state.rows[i];
			out.SetValue(0, i, Value::BIGINT(r.counter_id));
			out.SetValue(1, i, Value::BIGINT(r.cnt));
			out.SetValue(2, i, Value::DOUBLE(r.avg_len));
		}
	} else {
		out.SetCardinality(0);
	}

	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
//  Register
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
