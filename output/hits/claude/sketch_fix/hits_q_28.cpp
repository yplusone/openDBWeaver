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
#include <unordered_map>
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

	std::unordered_map<int64_t, AggState> agg_map;
	SortState sort_state;

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	std::unordered_map<int64_t, AggState> agg_map;
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
	// expected output order (as your expected sample shows): CounterID, c, l
	return_types.emplace_back(LogicalType::BIGINT); // CounterID
	return_types.emplace_back(LogicalType::BIGINT); // c
	return_types.emplace_back(LogicalType::DOUBLE); // l

	names.emplace_back("CounterID");
	names.emplace_back("c");
	names.emplace_back("l");

	return make_uniq<FnBindData>();
}

// ============================================================
//  Helpers: read CounterID robustly by physical type
// ============================================================

template <class T>
static inline int64_t LoadIntToI64(const data_ptr_t base, idx_t idx) {
	const auto ptr = reinterpret_cast<const T *>(base);
	return (int64_t)ptr[idx];
}

static inline int64_t LoadCounterID(const Vector &counter_vec,
                                   const UnifiedVectorFormat &counter_uvf,
                                   idx_t row_idx) {
	const idx_t i_ctr = counter_uvf.sel->get_index(row_idx);

	switch (counter_vec.GetType().id()) {
	case LogicalTypeId::TINYINT:
		return LoadIntToI64<int8_t>(counter_uvf.data, i_ctr);
	case LogicalTypeId::SMALLINT:
		return LoadIntToI64<int16_t>(counter_uvf.data, i_ctr);
	case LogicalTypeId::INTEGER:
		return LoadIntToI64<int32_t>(counter_uvf.data, i_ctr);
	case LogicalTypeId::BIGINT:
		return LoadIntToI64<int64_t>(counter_uvf.data, i_ctr);
	case LogicalTypeId::UTINYINT:
		return LoadIntToI64<uint8_t>(counter_uvf.data, i_ctr);
	case LogicalTypeId::USMALLINT:
		return LoadIntToI64<uint16_t>(counter_uvf.data, i_ctr);
	case LogicalTypeId::UINTEGER:
		return LoadIntToI64<uint32_t>(counter_uvf.data, i_ctr);
	case LogicalTypeId::UBIGINT:
		// may overflow int64, but ClickBench CounterID typically fits
		return (int64_t)LoadIntToI64<uint64_t>(counter_uvf.data, i_ctr);
	default:
		// be strict: unknown type -> throw
		throw InvalidInputException("Unsupported CounterID type: %s", counter_vec.GetType().ToString());
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

	D_ASSERT(input.ColumnCount() == 2);

	UnifiedVectorFormat url_uvf;
	UnifiedVectorFormat counter_uvf;

	input.data[0].ToUnifiedFormat(input.size(), url_uvf);      // URL
	input.data[1].ToUnifiedFormat(input.size(), counter_uvf);  // CounterID

	auto url_ptr = (string_t *)url_uvf.data;

	auto &valid_url = url_uvf.validity;
	auto &valid_counter = counter_uvf.validity;

	const bool url_all_valid = valid_url.AllValid();
	const bool counter_all_valid = valid_counter.AllValid();

	const idx_t n = input.size();

	// Fast branch: no NULLs
	if (url_all_valid && counter_all_valid) {
		for (idx_t row_idx = 0; row_idx < n; ++row_idx) {
			const idx_t i_url = url_uvf.sel->get_index(row_idx);

			const string_t url = url_ptr[i_url];
			const idx_t url_len = url.GetSize();
			if (url_len == 0) {
				continue; // WHERE URL <> ''
			}

			const int64_t counter_id = LoadCounterID(input.data[1], counter_uvf, row_idx);
			auto &st = l.agg_map[counter_id];
			st.sum_len += (__int128)url_len;
			st.cnt += 1;
		}
	} else {
		// NULL-safe branch
		for (idx_t row_idx = 0; row_idx < n; ++row_idx) {
			const idx_t i_url = url_uvf.sel->get_index(row_idx);
			const idx_t i_ctr = counter_uvf.sel->get_index(row_idx);

			if (!url_all_valid && !valid_url.RowIsValid(i_url)) {
				continue;
			}
			if (!counter_all_valid && !valid_counter.RowIsValid(i_ctr)) {
				continue;
			}

			const string_t url = url_ptr[i_url];
			const idx_t url_len = url.GetSize();
			if (url_len == 0) {
				continue; // WHERE URL <> ''
			}

			const int64_t counter_id = LoadCounterID(input.data[1], counter_uvf, row_idx);
			auto &st = l.agg_map[counter_id];
			st.sum_len += (__int128)url_len;
			st.cnt += 1;
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize: merge local maps; last finisher sorts + emits top-25
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
	auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
	auto &l = in.local_state->Cast<FnLocalState>();

	// merge exactly once per local state
	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			for (auto &kv : l.agg_map) {
				auto &dst = g.agg_map[kv.first];
				dst.sum_len += kv.second.sum_len;
				dst.cnt += kv.second.cnt;
			}
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	// only the last merged local emits output
	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active > 0 && merged == active) {
		// build rows: HAVING COUNT(*) > 100000
		{
			std::lock_guard<std::mutex> guard(g.lock);
			g.sort_state.rows.clear();
			g.sort_state.sorted = false;
			g.sort_state.rows.reserve(g.agg_map.size());

			for (auto &kv : g.agg_map) {
				const int64_t counter_id = kv.first;
				const auto &st = kv.second;
				if (st.cnt <= 100000) {
					continue;
				}
				const double avg_len = (double)st.sum_len / (double)st.cnt;
				g.sort_state.rows.push_back(SortRow{counter_id, st.cnt, avg_len});
			}
		}

		g.sort_state.SortNow();

		const idx_t out_n = MinValue<idx_t>((idx_t)g.sort_state.rows.size(), 25);
		out.SetCardinality(out_n);

		// output order: CounterID, c, l
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