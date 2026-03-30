/*
query_template:
SELECT
  REGEXP_REPLACE(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k,
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
  REGEXP_REPLACE(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k,
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
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/types/string_type.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>

namespace duckdb {

// ------------------------------
// Helper: match check for
//   ^https?://(?:www\.)?([^/]+)/.*$
// BUT replacement is treated as LITERAL "\\1" to match your DuckDB output.
// If not matched, return original string.
// ------------------------------
static inline bool StartsWith(const char *p, idx_t n, const char *lit) {
	for (idx_t i = 0; lit[i]; i++) {
		if (i >= n) return false;
		if (p[i] != lit[i]) return false;
	}
	return true;
}

static inline bool MatchesPattern_HasHostAndSlash(const char *p, idx_t n) {
	// ^https?://
	idx_t off = 0;
	if (StartsWith(p, n, "http://")) {
		off = 7;
	} else if (StartsWith(p, n, "https://")) {
		off = 8;
	} else {
		return false;
	}

	// (?:www\.)?
	if (off + 4 <= n && p[off] == 'w' && p[off + 1] == 'w' && p[off + 2] == 'w' && p[off + 3] == '.') {
		off += 4;
	}
	if (off >= n) return false;

	// ([^/]+)
	idx_t host_start = off;
	while (off < n && p[off] != '/') off++;
	idx_t host_len = off - host_start;

	// must have "/.*$" => require a '/' after host
	if (off >= n || p[off] != '/') return false;
	if (host_len == 0) return false;

	return true;
}

static inline std::string ComputeK_LiteralBackrefBehavior(const string_t &referer) {
	const auto n = referer.GetSize();
	const char *p = referer.GetData();
	if (MatchesPattern_HasHostAndSlash(p, n)) {
		// literal "\1" (two chars)
		return std::string("\\1");
	}
	// not matched => original string
	return referer.GetString();
}

// ------------------------------
// Per-group aggregation
// ------------------------------
struct GroupAgg {
	int64_t cnt = 0;        // match DuckDB COUNT(*) output type int64
	uint64_t sum_len = 0;   // sum(strlen)
	string_t min_ref;       // store minimal referer (as string_t, not std::string)
	bool has_min = false;

	inline void Update(const string_t &ref) {
		cnt++;
		sum_len += (uint64_t)ref.GetSize();
		if (!has_min) {
			min_ref = ref;
			has_min = true;
		} else {
			// Compare string_t by binary content
			if (ref.GetSize() < min_ref.GetSize() ||
				(ref.GetSize() == min_ref.GetSize() && memcmp(ref.GetData(), min_ref.GetData(), ref.GetSize()) < 0)) {
				min_ref = ref;
			}
		}
	}
};
struct ResultRow {
	std::string k;
	double l = 0.0;
	int64_t c = 0;
	string_t min_referer;
};


// ------------------------------
// Bind / states
// ------------------------------
struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};
struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;
	unordered_map<std::string, GroupAgg> groups;
	bool reserved_global_groups = false;
	idx_t estimated_group_count = 0;


	bool result_built = false;
	std::vector<ResultRow> result;
	idx_t result_pos = 0;

	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
	unordered_map<std::string, GroupAgg> local_groups;
	bool merged = false;
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);

	auto l = make_uniq<FnLocalState>();
	l->local_groups.reserve(1 << 18);
	return std::move(l);
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types, vector<string> &names) {
	// Match your base query output types:
	// k varchar, l double, c int64, min_referer varchar
	return_types.emplace_back(LogicalType::VARCHAR);
	return_types.emplace_back(LogicalType::DOUBLE);
	return_types.emplace_back(LogicalType::BIGINT);
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("k");
	names.emplace_back("l");
	names.emplace_back("c");
	names.emplace_back("min_referer");

	return make_uniq<FnBindData>();
}

// ------------------------------
// Execute
// ------------------------------
static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	UnifiedVectorFormat ref_uvf;
	input.data[0].ToUnifiedFormat(input.size(), ref_uvf);
	auto ref_ptr = (string_t *)ref_uvf.data;
	auto &valid_ref = ref_uvf.validity;
	const bool all_valid = valid_ref.AllValid();
	const VectorType ref_type = input.data[0].GetVectorType();
if (ref_type == VectorType::CONSTANT_VECTOR) {
		const idx_t ridx0 = ref_uvf.sel->get_index(0);
		if (!valid_ref.RowIsValid(ridx0)) return OperatorResultType::NEED_MORE_INPUT;

		const auto s = ref_ptr[ridx0];
		if (s.GetSize() == 0) return OperatorResultType::NEED_MORE_INPUT; // WHERE Referer <> ''

		const char *p = s.GetData();
		idx_t n = s.GetSize();
		std::string k = MatchesPattern_HasHostAndSlash(p, n) ? std::string("\\1") : std::string(p, n);

		auto &st = l.local_groups[k];
		st.cnt += (int64_t)input.size();
		st.sum_len += (uint64_t)n * (uint64_t)input.size();
		if (!st.has_min) {
			st.min_ref = s;
			st.has_min = true;
		} else if (s.GetSize() < st.min_ref.GetSize() ||
			(s.GetSize() == st.min_ref.GetSize() && memcmp(s.GetData(), st.min_ref.GetData(), s.GetSize()) < 0)) {
			st.min_ref = s;
		}
		return OperatorResultType::NEED_MORE_INPUT;
	}
if (all_valid) {
		for (idx_t row = 0; row < input.size(); row++) {
			const idx_t ridx = ref_uvf.sel->get_index(row);
			const auto s = ref_ptr[ridx];
			if (s.GetSize() == 0) continue;

			const char *p = s.GetData();
			idx_t n = s.GetSize();
			std::string k = MatchesPattern_HasHostAndSlash(p, n) ? std::string("\\1") : std::string(p, n);

			auto &st = l.local_groups[k];
			st.Update(s);
		}
	} else {
		for (idx_t row = 0; row < input.size(); row++) {
			const idx_t ridx = ref_uvf.sel->get_index(row);
			if (!valid_ref.RowIsValid(ridx)) continue;

			const auto s = ref_ptr[ridx];
			if (s.GetSize() == 0) continue;

			const char *p = s.GetData();
			idx_t n = s.GetSize();
			std::string k = MatchesPattern_HasHostAndSlash(p, n) ? std::string("\\1") : std::string(p, n);

			auto &st = l.local_groups[k];
			st.Update(s);
		}
	}


	return OperatorResultType::NEED_MORE_INPUT;
}

static void BuildFinalTopK(FnGlobalState &g) {
	if (g.result_built) return;

	std::vector<ResultRow> tmp;
	tmp.reserve(g.groups.size());
for (auto &it : g.groups) {
		auto &k = it.first;
		auto &st = it.second;

		// HAVING COUNT(*) > 100000
		if (st.cnt <= 100000) continue;

		ResultRow r;
		r.k = k;
		r.c = st.cnt;
		r.l = (st.cnt > 0) ? (double)st.sum_len / (double)st.cnt : 0.0;
		r.min_referer = st.has_min ? st.min_ref : string_t();
		tmp.push_back(std::move(r));
	}


	// ORDER BY l DESC LIMIT 25
	std::sort(tmp.begin(), tmp.end(), [](const ResultRow &a, const ResultRow &b) {
		if (a.l != b.l) return a.l > b.l;
		if (a.c != b.c) return a.c > b.c;
		return a.k < b.k;
	});
	if (tmp.size() > 25) tmp.resize(25);

	g.result = std::move(tmp);
	g.result_pos = 0;
	g.result_built = true;
}
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
	auto &l = in.local_state->Cast<FnLocalState>();
	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			// Estimate total expected unique groups on first merge, inside the lock
			if (!g.reserved_global_groups) {
				idx_t estimated_groups = l.local_groups.size() * g.active_local_states.load(std::memory_order_relaxed);
				// Clamp to a minimum value in case l.local_groups is tiny
				if (estimated_groups < (1 << 16)) {
					estimated_groups = (1 << 16);
				}
				g.groups.reserve(estimated_groups);
				g.estimated_group_count = estimated_groups;
				g.reserved_global_groups = true;
			}

			for (auto &it : l.local_groups) {
				auto &k = it.first;
				auto &src = it.second;
				auto &dst = g.groups[k];
				dst.cnt += src.cnt;
				dst.sum_len += src.sum_len;
				if (src.has_min) {
					if (!dst.has_min || src.min_ref < dst.min_ref) {
						dst.min_ref = src.min_ref;
						dst.has_min = true;
					}
				}
			}
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}


	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	const auto active = g.active_local_states.load(std::memory_order_relaxed);
	if (!(active > 0 && merged == active)) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	{
		std::lock_guard<std::mutex> guard(g.lock);
		BuildFinalTopK(g);
	}

	out.Reset();
	idx_t remaining = g.result.size() - g.result_pos;
	idx_t nout = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
	if (nout == 0) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	out.SetCardinality(nout);

	auto &vk = out.data[0];
	auto pk = FlatVector::GetData<string_t>(vk);
	auto pl = FlatVector::GetData<double>(out.data[1]);
	auto pc = FlatVector::GetData<int64_t>(out.data[2]);
	auto &vmin = out.data[3];
	auto pmin = FlatVector::GetData<string_t>(vmin);
for (idx_t i = 0; i < nout; i++) {
		auto &row = g.result[g.result_pos + i];
		pk[i] = StringVector::AddString(vk, row.k);
		pl[i] = row.l;
		pc[i] = row.c;
		if (row.min_referer.GetSize() > 0)
			pmin[i] = StringVector::AddString(vmin, row.min_referer.GetData(), row.min_referer.GetSize());
		else
			pmin[i] = StringVector::AddString(vmin, "", 0);
	}


	g.result_pos += nout;
	return OperatorFinalizeResultType::FINISHED;
}

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
	f.in_out_function       = FnExecute;
	f.in_out_function_final = FnFinalize;
	loader.RegisterFunction(f);
}

void DbweaverExtension::Load(ExtensionLoader &loader) { LoadInternal(loader); }
std::string DbweaverExtension::Name() { return "dbweaver"; }
std::string DbweaverExtension::Version() const { return DuckDB::LibraryVersion(); }

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) { duckdb::LoadInternal(loader); }
}
