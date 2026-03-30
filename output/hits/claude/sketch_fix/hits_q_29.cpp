/*
query_template:
SELECT
  REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\\1') AS k,
  AVG(STRLEN(Referer)) AS l,
  COUNT(*) AS c,
  MIN(Referer) AS min_referer
FROM hits
WHERE Referer <> ''
GROUP BY k
HAVING COUNT(*) > 100000
ORDER BY l DESC
LIMIT 25;

split_template:
SELECT k, l, c, min_referer
FROM dbweaver((
  SELECT Referer
  FROM hits
));

query_example:
SELECT
  REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k,
  AVG(STRLEN(Referer)) AS l,
  COUNT(*) AS c,
  MIN(Referer) AS min_referer
FROM hits
WHERE Referer <> ''
GROUP BY k
HAVING COUNT(*) > 100000
ORDER BY l DESC
LIMIT 25;

split_query:
SELECT k, l, c, min_referer
FROM dbweaver((
  SELECT Referer
  FROM hits
));
*/
/*
Baseline (unoptimized) implementation:
- Per-row RE2::FullMatch (no fast-path parsing)
- No special-case handling for '\n' / '\0' / invalid UTF-8
- STRLEN uses byte length (string_t::GetSize)
- Uses remaining_local_states gate to ensure finalize outputs exactly once (no hang)
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"

#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdint>

// DuckDB vendored RE2
#include "re2/re2.h"
using duckdb_re2::RE2;
using duckdb_re2::StringPiece;

namespace duckdb {

// ============================================================
// Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

// ============================================================
// Regex helper (naive)
// ============================================================

// Pattern: '^https?://(?:www\\.)?([^/]+)/.*$'
static inline const RE2 &GetKRegex() {
	static const RE2 re("^https?://(?:www\\.)?([^/]+)/.*$");
	return re;
}

// Naive semantics: if FullMatch -> return group1 else return original
static inline std::string ExtractK_Naive(const string_t &ref) {
	const char *s = ref.GetData();
	const idx_t n = ref.GetSize();
	if (n == 0) return std::string();

	StringPiece text(s, (size_t)n);
	StringPiece g1;
	if (RE2::FullMatch(text, GetKRegex(), &g1)) {
		return std::string(g1.data(), g1.size());
	}
	return ref.GetString();
}

// ============================================================
// Aggregation structures
// ============================================================

struct GroupKey {
	std::string k;
	bool operator==(const GroupKey &other) const { return k == other.k; }
};

struct GroupKeyHash {
	size_t operator()(const GroupKey &g) const { return std::hash<std::string>()(g.k); }
};

struct AggState {
	int64_t count = 0;
	int64_t sum_len = 0;     // SUM(STRLEN(Referer)) -> byte length
	std::string min_ref;     // MIN(Referer)
	bool initialized = false;
};

struct SortRow {
	GroupKey key;
	AggState state;
	double l; // AVG
};

struct SortRowComparator {
	bool operator()(const SortRow &a, const SortRow &b) const {
		// ORDER BY l DESC
		return a.l > b.l;
	}
};

struct SortState {
	std::vector<SortRow> rows;
	bool sorted = false;

	inline void AddRow(const GroupKey &key, const AggState &state) {
		const double avg = state.count ? (double)state.sum_len / (double)state.count : 0.0;
		rows.push_back(SortRow{key, state, avg});
	}

	inline void SortNow() {
		if (!sorted) {
			std::sort(rows.begin(), rows.end(), SortRowComparator{});
			sorted = true;
		}
	}
};

// ============================================================
// Global/Local states
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;

	// Correct finalize gating (prevents "never output" hang)
	std::atomic<idx_t> remaining_local_states{0};

	std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
	SortState sort_state;

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	(void)GetKRegex();
	return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.remaining_local_states.fetch_add(1, std::memory_order_relaxed);
	return make_uniq<FnLocalState>();
}

// ============================================================
// Bind
// ============================================================

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	names.push_back("k");
	names.push_back("l");
	names.push_back("c");
	names.push_back("min_referer");

	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::DOUBLE);
	return_types.push_back(LogicalType::BIGINT);
	return_types.push_back(LogicalType::VARCHAR);

	return make_uniq<FnBindData>();
}

// ============================================================
// Execute (aggregate in local map)
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	UnifiedVectorFormat ref_uvf;
	input.data[0].ToUnifiedFormat(input.size(), ref_uvf);

	auto &valid_ref = ref_uvf.validity;
	const bool all_valid = valid_ref.AllValid();

	string_t *ref_ptr = (string_t *)ref_uvf.data;
	const idx_t nrows = input.size();

	// WHERE Referer <> ''
	if (all_valid) {
		for (idx_t row = 0; row < nrows; ++row) {
			const idx_t i = ref_uvf.sel->get_index(row);
			const string_t vref = ref_ptr[i];

			const idx_t ref_len = vref.GetSize(); // STRLEN byte length
			if (ref_len == 0) continue;

			GroupKey key;
			key.k = ExtractK_Naive(vref);

			auto &agg = l.agg_map[key];
			if (!agg.initialized) {
				agg.count = 1;
				agg.sum_len = (int64_t)ref_len;
				agg.min_ref = vref.GetString();
				agg.initialized = true;
			} else {
				agg.count++;
				agg.sum_len += (int64_t)ref_len;
				const std::string cur = vref.GetString();
				if (cur < agg.min_ref) agg.min_ref = cur;
			}
		}
	} else {
		for (idx_t row = 0; row < nrows; ++row) {
			const idx_t i = ref_uvf.sel->get_index(row);
			if (!valid_ref.RowIsValid(i)) continue;

			const string_t vref = ref_ptr[i];
			const idx_t ref_len = vref.GetSize();
			if (ref_len == 0) continue;

			GroupKey key;
			key.k = ExtractK_Naive(vref);

			auto &agg = l.agg_map[key];
			if (!agg.initialized) {
				agg.count = 1;
				agg.sum_len = (int64_t)ref_len;
				agg.min_ref = vref.GetString();
				agg.initialized = true;
			} else {
				agg.count++;
				agg.sum_len += (int64_t)ref_len;
				const std::string cur = vref.GetString();
				if (cur < agg.min_ref) agg.min_ref = cur;
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
// Finalize (merge local, last finalize outputs results)
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
	auto &l = in.local_state->Cast<FnLocalState>();

	// Merge local -> global once
	if (!l.merged) {
		std::lock_guard<std::mutex> guard(g.lock);
		for (const auto &entry : l.agg_map) {
			const GroupKey &key = entry.first;
			const AggState &st = entry.second;

			auto &gst = g.agg_map[key];
			if (!gst.initialized) {
				gst = st;
			} else {
				gst.count += st.count;
				gst.sum_len += st.sum_len;
				if (st.min_ref < gst.min_ref) gst.min_ref = st.min_ref;
			}
		}
		l.merged = true;
	}

	// Last finalize emits output
	const idx_t prev = g.remaining_local_states.fetch_sub(1, std::memory_order_acq_rel);
	if (prev == 1) {
		std::lock_guard<std::mutex> guard(g.lock);

		// HAVING COUNT(*) > 100000
		for (const auto &entry : g.agg_map) {
			const auto &key = entry.first;
			const auto &st = entry.second;
			if (st.count > 100000) {
				g.sort_state.AddRow(key, st);
			}
		}

		// ORDER BY l DESC, LIMIT 25
		g.sort_state.SortNow();

		idx_t out_idx = 0;
		const size_t max_out = std::min<size_t>(25, g.sort_state.rows.size());
		for (size_t i = 0; i < max_out; ++i) {
			const auto &row = g.sort_state.rows[i];
			out.SetValue(0, out_idx, Value(row.key.k));
			out.SetValue(1, out_idx, Value(row.l));
			out.SetValue(2, out_idx, Value::BIGINT(row.state.count));
			out.SetValue(3, out_idx, Value(row.state.min_ref));
			out_idx++;
		}
		out.SetCardinality(out_idx);
	} else {
		out.SetCardinality(0);
	}

	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
// Register
// ============================================================

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
	f.in_out_function = FnExecute;
	f.in_out_function_final = FnFinalize;
	loader.RegisterFunction(f);
}

void DbweaverExtension::Load(ExtensionLoader &loader) { LoadInternal(loader); }
std::string DbweaverExtension::Name() { return "dbweaver"; }
std::string DbweaverExtension::Version() const { return DuckDB::LibraryVersion(); }

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) {
	duckdb::LoadInternal(loader);
}

}