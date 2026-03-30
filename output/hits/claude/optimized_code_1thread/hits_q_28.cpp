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
	uint64_t sum_len = 0; // SUM(STRLEN(URL))
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

static constexpr size_t INITIAL_CAPACITY = 300000;

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;
	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};

	std::vector<AggState> agg_vec;
	std::vector<uint32_t> active_ids;
	SortState sort_state;

	FnGlobalState() {
		agg_vec.resize(INITIAL_CAPACITY);
		active_ids.reserve(16384);
	}

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	std::vector<AggState> agg_vec;
	std::vector<uint32_t> active_ids;

	FnLocalState() {
		agg_vec.resize(INITIAL_CAPACITY);
		active_ids.reserve(16384);
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
	g.active_local_states.fetch_add(1);
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
//  Execute: ProcessRowsTyped hoisted out of the main loop
// ============================================================
template <class T>
static void ProcessRowsTyped(std::vector<AggState> &agg_vec, 
                             std::vector<uint32_t> &active_ids,
                             const UnifiedVectorFormat &url_uvf, 
                             const UnifiedVectorFormat &counter_uvf, 
                             idx_t n) {
	auto url_ptr = (string_t *)url_uvf.data;
	auto counter_ptr = (T *)counter_uvf.data;
	const bool counter_all_valid = counter_uvf.validity.AllValid();

	for (idx_t row_idx = 0; row_idx < n; ++row_idx) {
		const idx_t i_ctr = counter_uvf.sel->get_index(row_idx);
		if (!counter_all_valid && !counter_uvf.validity.RowIsValid(i_ctr)) continue;

		const int64_t counter_id = (int64_t)counter_ptr[i_ctr];
		if (DUCKDB_UNLIKELY(counter_id < 0)) continue;
		if (DUCKDB_UNLIKELY((size_t)counter_id >= agg_vec.size())) {
			agg_vec.resize(counter_id + 100000);
		}
		AggState &st = agg_vec[counter_id];
		if (st.cnt == 0) {
			active_ids.push_back((uint32_t)counter_id);
		}
		const idx_t i_url = url_uvf.sel->get_index(row_idx);
		st.sum_len += (uint64_t)url_ptr[i_url].GetSize();
		st.cnt += 1;
	}
}


static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                   DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	UnifiedVectorFormat url_uvf, counter_uvf;
	input.data[0].ToUnifiedFormat(input.size(), url_uvf);
	input.data[1].ToUnifiedFormat(input.size(), counter_uvf);

	const idx_t n = input.size();

	switch (input.data[1].GetType().id()) {
	case LogicalTypeId::TINYINT:
		ProcessRowsTyped<int8_t>(l.agg_vec, l.active_ids, url_uvf, counter_uvf, n);
		break;
	case LogicalTypeId::SMALLINT:
		ProcessRowsTyped<int16_t>(l.agg_vec, l.active_ids, url_uvf, counter_uvf, n);
		break;
	case LogicalTypeId::INTEGER:
		ProcessRowsTyped<int32_t>(l.agg_vec, l.active_ids, url_uvf, counter_uvf, n);
		break;
	case LogicalTypeId::BIGINT:
		ProcessRowsTyped<int64_t>(l.agg_vec, l.active_ids, url_uvf, counter_uvf, n);
		break;
	case LogicalTypeId::UTINYINT:
		ProcessRowsTyped<uint8_t>(l.agg_vec, l.active_ids, url_uvf, counter_uvf, n);
		break;
	case LogicalTypeId::USMALLINT:
		ProcessRowsTyped<uint16_t>(l.agg_vec, l.active_ids, url_uvf, counter_uvf, n);
		break;
	case LogicalTypeId::UINTEGER:
		ProcessRowsTyped<uint32_t>(l.agg_vec, l.active_ids, url_uvf, counter_uvf, n);
		break;
	case LogicalTypeId::UBIGINT:
		ProcessRowsTyped<uint64_t>(l.agg_vec, l.active_ids, url_uvf, counter_uvf, n);
		break;
	default:
		throw InvalidInputException("Unsupported CounterID type: %s", input.data[1].GetType().ToString());
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize: merge local vectors; last finisher sorts + emits
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			for (auto cid : l.active_ids) {
				if (cid >= g.agg_vec.size()) {
					g.agg_vec.resize(cid + 100000);
				}
				if (g.agg_vec[cid].cnt == 0) {
					g.active_ids.push_back(cid);
				}
				g.agg_vec[cid].sum_len += l.agg_vec[cid].sum_len;
				g.agg_vec[cid].cnt += l.agg_vec[cid].cnt;
			}
		}
		l.merged = true;
		if (g.merged_local_states.fetch_add(1) + 1 == g.active_local_states.load()) {
			{
				std::lock_guard<std::mutex> guard(g.lock);
				if (!g.sort_state.sorted) {
					g.sort_state.rows.clear();
					for (auto cid : g.active_ids) {
						if (g.agg_vec[cid].cnt > 100000) {
							const double avg_len = (double)g.agg_vec[cid].sum_len / (double)g.agg_vec[cid].cnt;
							g.sort_state.rows.push_back(SortRow{(int64_t)cid, g.agg_vec[cid].cnt, avg_len});
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
	} else {
		out.SetCardinality(0);
	}

	return OperatorFinalizeResultType::FINISHED;
}

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
