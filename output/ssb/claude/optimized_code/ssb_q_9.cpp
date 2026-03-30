/*
query_template: SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE C_CITY IN (:c_city_1, :c_city_2)
  AND S_CITY IN (:s_city_1, :s_city_2)
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN :year_lo AND :year_hi
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC

split_template: SELECT C_CITY, S_CITY, year, revenue
FROM dbweaver((
  SELECT C_CITY, S_CITY, LO_ORDERDATE, LO_REVENUE
  FROM lineorder_flat WHERE (C_CITY = :c_city_1 OR C_CITY = :c_city_2) AND (S_CITY = :s_city_1 OR S_CITY = :s_city_2)
), :year_lo, :year_hi) ORDER BY year ASC, revenue DESC;

query_example: SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE
  (C_CITY = 'UNITED KI1' OR C_CITY = 'UNITED KI5')
  AND (S_CITY = 'UNITED KI1' OR S_CITY = 'UNITED KI5')
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN 1992 AND 1997
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC

split_query: SELECT C_CITY, S_CITY, year, revenue
FROM dbweaver((
  SELECT C_CITY, S_CITY, LO_ORDERDATE, LO_REVENUE
  FROM lineorder_flat WHERE (C_CITY = 'UNITED KI1' OR C_CITY = 'UNITED KI5') AND (S_CITY = 'UNITED KI1' OR S_CITY = 'UNITED KI5')
), 1992, 1997) ORDER BY year ASC, revenue DESC;
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"

#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace duckdb {

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

static constexpr int32_t Y1992_START_DAYS = 8035;
static constexpr int32_t Y1993_START_DAYS = 8401;
static constexpr int32_t Y1994_START_DAYS = 8766;
static constexpr int32_t Y1995_START_DAYS = 9131;
static constexpr int32_t Y1996_START_DAYS = 9496;
static constexpr int32_t Y1997_START_DAYS = 9862;
static constexpr int32_t Y1998_START_DAYS = 10227;

static int32_t ExtractYearFromDate(date_t date) {
	const int32_t days = date.days;
	if (days >= Y1998_START_DAYS) return 1998;
	if (days >= Y1997_START_DAYS) return 1997;
	if (days >= Y1996_START_DAYS) return 1996;
	if (days >= Y1995_START_DAYS) return 1995;
	if (days >= Y1994_START_DAYS) return 1994;
	if (days >= Y1993_START_DAYS) return 1993;
	if (days >= Y1992_START_DAYS) return 1992;

	// Fallback for dates earlier than 1992
	int32_t year, month, day;
	Date::Convert(date, year, month, day);
	return year;
}

// strict weak ordering for string_t (lexicographic by bytes, then length)
static inline bool StringTLess(const string_t &a, const string_t &b) {
	const auto asz = a.GetSize();
	const auto bsz = b.GetSize();
	const auto n = std::min(asz, bsz);
	int cmp = 0;
	if (n > 0) {
		cmp = std::memcmp(a.GetData(), b.GetData(), n);
	}
	if (cmp != 0) {
		return cmp < 0;
	}
	return asz < bsz;
}

// Map a value into {0,1}. The predicate guarantees <=2 distinct values.
// We "learn" the first two values observed in this local state.
static inline uint8_t Map2(const string_t &v, string_t vals[2], uint8_t &cnt) {
	if (cnt > 0 && v == vals[0]) return 0;
	if (cnt > 1 && v == vals[1]) return 1;
	if (cnt < 2) {
		vals[cnt] = v;
		return cnt++;
	}
	// Should not happen under the query predicate, but keep a safe fallback.
	return 0;
}

// Canonicalize local (c_vals, s_vals) ordering so different threads merge consistently.
// If we swap C indices: bucket cs (c*2+s) must swap 0<->2 and 1<->3 for each year.
// If we swap S indices: bucket cs must swap 0<->1 and 2<->3 for each year.
static inline void CanonicalizeLocal(string_t c_vals[2], uint8_t c_cnt,
                                    string_t s_vals[2], uint8_t s_cnt,
                                    vector<__int128> &sums, vector<uint32_t> &counts,
                                    idx_t years) {
	if (years == 0) return;

	const auto swap_c = (c_cnt == 2 && StringTLess(c_vals[1], c_vals[0]));
	const auto swap_s = (s_cnt == 2 && StringTLess(s_vals[1], s_vals[0]));

	if (swap_c) {
		std::swap(c_vals[0], c_vals[1]);
		for (idx_t y = 0; y < years; ++y) {
			const idx_t base = y * 4;
			std::swap(sums[base + 0], sums[base + 2]);
			std::swap(sums[base + 1], sums[base + 3]);
			std::swap(counts[base + 0], counts[base + 2]);
			std::swap(counts[base + 1], counts[base + 3]);
		}
	}

	if (swap_s) {
		std::swap(s_vals[0], s_vals[1]);
		for (idx_t y = 0; y < years; ++y) {
			const idx_t base = y * 4;
			std::swap(sums[base + 0], sums[base + 1]);
			std::swap(sums[base + 2], sums[base + 3]);
			std::swap(counts[base + 0], counts[base + 1]);
			std::swap(counts[base + 2], counts[base + 3]);
		}
	}
}

// Convert __int128 accumulator to DuckDB hugeint_t
static inline hugeint_t ToHugeint(__int128 acc) {
	hugeint_t result;
	result.lower = static_cast<uint64_t>(acc);
	result.upper = static_cast<int64_t>(acc >> 64);
	return result;
}

// ------------------------------------------------------------
// Bind / State
// ------------------------------------------------------------

struct FnBindData : public FunctionData {
	int32_t year_lo;
	int32_t year_hi;
	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<FnBindData>();
		copy->year_lo = year_lo;
		copy->year_hi = year_hi;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other) const override {
		auto &cast = other.Cast<FnBindData>();
		return year_lo == cast.year_lo && year_hi == cast.year_hi;
	}
};

struct FnGlobalState : public GlobalTableFunctionState {
	// Global buckets: years * 4
	vector<__int128> sums;
	vector<uint32_t> counts;

	// Canonical city values (lexicographic order). Filled from the first merged local.
	string_t c_vals[2];
	uint8_t c_cnt = 0;
	string_t s_vals[2];
	uint8_t s_cnt = 0;

	bool inited = false;

	std::mutex lock;
	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};
	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }

	void InitIfNeeded(idx_t years) {
		if (inited) return;
		sums.assign(years * 4, 0);
		counts.assign(years * 4, 0);
		inited = true;
	}
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
	// Learned local mapping values
	string_t c_vals[2];
	uint8_t c_cnt = 0;
	string_t s_vals[2];
	uint8_t s_cnt = 0;

	// Local buckets: years * 4
	vector<__int128> sums;
	vector<uint32_t> counts;

	bool inited = false;
	bool merged = false;

	void InitIfNeeded(idx_t years) {
		if (inited) return;
		sums.assign(years * 4, 0);
		counts.assign(years * 4, 0);
		inited = true;
	}
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                       GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);
	return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<string> &names) {
	auto year_lo = input.inputs[1].GetValue<int32_t>();
	auto year_hi = input.inputs[2].GetValue<int32_t>();

	return_types.emplace_back(LogicalType::VARCHAR); // C_CITY
	return_types.emplace_back(LogicalType::VARCHAR); // S_CITY
	return_types.emplace_back(LogicalType::BIGINT);  // year
	return_types.emplace_back(LogicalType::HUGEINT); // revenue

	names.emplace_back("C_CITY");
	names.emplace_back("S_CITY");
	names.emplace_back("year");
	names.emplace_back("revenue");

	auto bind_data = make_uniq<FnBindData>();
	bind_data->year_lo = year_lo;
	bind_data->year_hi = year_hi;
	return std::move(bind_data);
}

// ------------------------------------------------------------
// Execute
// ------------------------------------------------------------

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();
	auto &bind_data = in.bind_data->Cast<FnBindData>();
	const int32_t year_lo = bind_data.year_lo;
	const int32_t year_hi = bind_data.year_hi;

	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	const idx_t years = (year_hi >= year_lo) ? (idx_t)(year_hi - year_lo + 1) : 0;
	l.InitIfNeeded(years);

	UnifiedVectorFormat c_city_uvf;
	input.data[0].ToUnifiedFormat(input.size(), c_city_uvf);
	auto *c_city_ptr = reinterpret_cast<string_t *>(c_city_uvf.data);

	UnifiedVectorFormat s_city_uvf;
	input.data[1].ToUnifiedFormat(input.size(), s_city_uvf);
	auto *s_city_ptr = reinterpret_cast<string_t *>(s_city_uvf.data);

	UnifiedVectorFormat lo_orderdate_uvf;
	input.data[2].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
	auto *lo_orderdate_ptr = reinterpret_cast<date_t *>(lo_orderdate_uvf.data);

	UnifiedVectorFormat lo_revenue_uvf;
	input.data[3].ToUnifiedFormat(input.size(), lo_revenue_uvf);
	auto *lo_revenue_ptr = reinterpret_cast<uint32_t *>(lo_revenue_uvf.data);

	auto &valid_c_city = c_city_uvf.validity;
	auto &valid_s_city = s_city_uvf.validity;
	auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
	auto &valid_lo_revenue = lo_revenue_uvf.validity;

	const bool c_city_all_valid = valid_c_city.AllValid();
	const bool s_city_all_valid = valid_s_city.AllValid();
	const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
	const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();

	const bool c_city_is_flat = (c_city_uvf.sel == nullptr);
	const bool s_city_is_flat = (s_city_uvf.sel == nullptr);
	const bool lo_orderdate_is_flat = (lo_orderdate_uvf.sel == nullptr);
	const bool lo_revenue_is_flat = (lo_revenue_uvf.sel == nullptr);

	const idx_t count = input.size();

	// Fastest path: all valid + no selection vectors
	if (c_city_all_valid && s_city_all_valid && lo_orderdate_all_valid && lo_revenue_all_valid &&
	    c_city_is_flat && s_city_is_flat && lo_orderdate_is_flat && lo_revenue_is_flat) {

		for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
			const string_t c_city_val = c_city_ptr[row_idx];
			const string_t s_city_val = s_city_ptr[row_idx];
			const date_t lo_orderdate_val = lo_orderdate_ptr[row_idx];
			const uint32_t lo_revenue_val = lo_revenue_ptr[row_idx];
			const int32_t y = ExtractYearFromDate(lo_orderdate_val);
			if (y < year_lo || y > year_hi) continue;
			const uint8_t c_idx = Map2(c_city_val, l.c_vals, l.c_cnt);
			const uint8_t s_idx = Map2(s_city_val, l.s_vals, l.s_cnt);

			const idx_t yy = (idx_t)(y - year_lo);
			const idx_t cs = (idx_t)(c_idx * 2 + s_idx);
			const idx_t idx = yy * 4 + cs;

			l.sums[idx] += lo_revenue_val;
			l.counts[idx] += 1;
		}

	} else if (c_city_all_valid && s_city_all_valid && lo_orderdate_all_valid && lo_revenue_all_valid) {

		for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
			const idx_t i_c_city = c_city_is_flat ? row_idx : c_city_uvf.sel->get_index(row_idx);
			const idx_t i_s_city = s_city_is_flat ? row_idx : s_city_uvf.sel->get_index(row_idx);
			const idx_t i_lo_orderdate = lo_orderdate_is_flat ? row_idx : lo_orderdate_uvf.sel->get_index(row_idx);
			const idx_t i_lo_revenue = lo_revenue_is_flat ? row_idx : lo_revenue_uvf.sel->get_index(row_idx);

			const string_t c_city_val = c_city_ptr[i_c_city];
			const string_t s_city_val = s_city_ptr[i_s_city];
			const date_t lo_orderdate_val = lo_orderdate_ptr[i_lo_orderdate];
			const uint32_t lo_revenue_val = lo_revenue_ptr[i_lo_revenue];
			const int32_t y = ExtractYearFromDate(lo_orderdate_val);
			if (y < year_lo || y > year_hi) continue;
			const uint8_t c_idx = Map2(c_city_val, l.c_vals, l.c_cnt);
			const uint8_t s_idx = Map2(s_city_val, l.s_vals, l.s_cnt);

			const idx_t yy = (idx_t)(y - year_lo);
			const idx_t cs = (idx_t)(c_idx * 2 + s_idx);
			const idx_t idx = yy * 4 + cs;

			l.sums[idx] += lo_revenue_val;
			l.counts[idx] += 1;
		}

	} else {

		for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
			const idx_t i_c_city = c_city_is_flat ? row_idx : c_city_uvf.sel->get_index(row_idx);
			const idx_t i_s_city = s_city_is_flat ? row_idx : s_city_uvf.sel->get_index(row_idx);
			const idx_t i_lo_orderdate = lo_orderdate_is_flat ? row_idx : lo_orderdate_uvf.sel->get_index(row_idx);
			const idx_t i_lo_revenue = lo_revenue_is_flat ? row_idx : lo_revenue_uvf.sel->get_index(row_idx);

			if (!c_city_all_valid && !valid_c_city.RowIsValid(i_c_city)) continue;
			if (!s_city_all_valid && !valid_s_city.RowIsValid(i_s_city)) continue;
			if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) continue;
			if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) continue;

			const string_t c_city_val = c_city_ptr[i_c_city];
			const string_t s_city_val = s_city_ptr[i_s_city];
			const date_t lo_orderdate_val = lo_orderdate_ptr[i_lo_orderdate];
			const uint32_t lo_revenue_val = lo_revenue_ptr[i_lo_revenue];

			const int32_t y = ExtractYearFromDate(lo_orderdate_val);
			if (y < year_lo || y > year_hi) continue;

			const uint8_t c_idx = Map2(c_city_val, l.c_vals, l.c_cnt);
			const uint8_t s_idx = Map2(s_city_val, l.s_vals, l.s_cnt);

			const idx_t yy = (idx_t)(y - year_lo);
			const idx_t cs = (idx_t)(c_idx * 2 + s_idx);
			const idx_t idx = yy * 4 + cs;

			l.sums[idx] += lo_revenue_val;
			l.counts[idx] += 1;
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ------------------------------------------------------------
// Finalize (merge locals -> output all groups once)
// ------------------------------------------------------------

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
	auto &l = in.local_state->Cast<FnLocalState>();
	auto &bind_data = in.bind_data->Cast<FnBindData>();
	const int32_t year_lo = bind_data.year_lo;
	const int32_t year_hi = bind_data.year_hi;

	const idx_t years = (year_hi >= year_lo) ? (idx_t)(year_hi - year_lo + 1) : 0;

	if (!l.merged) {
		// Make local ordering deterministic across threads before merging
		CanonicalizeLocal(l.c_vals, l.c_cnt, l.s_vals, l.s_cnt, l.sums, l.counts, years);

		{
			std::lock_guard<std::mutex> guard(g.lock);

			g.InitIfNeeded(years);

			// Initialize global canonical values from the first merged local
			if (g.c_cnt == 0 && l.c_cnt > 0) {
				g.c_vals[0] = l.c_vals[0];
				g.c_cnt = l.c_cnt;
				if (l.c_cnt == 2) g.c_vals[1] = l.c_vals[1];
			}
			if (g.s_cnt == 0 && l.s_cnt > 0) {
				g.s_vals[0] = l.s_vals[0];
				g.s_cnt = l.s_cnt;
				if (l.s_cnt == 2) g.s_vals[1] = l.s_vals[1];
			}

			// Merge buckets (same canonical order after CanonicalizeLocal)
			const idx_t n = years * 4;
			for (idx_t i = 0; i < n; ++i) {
				g.sums[i] += l.sums[i];
				g.counts[i] += l.counts[i];
			}
		}

		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	// Only the last local to merge emits the final result chunk.
	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active > 0 && merged == active) {
		// Collect all non-empty groups into output (typically <= 4 * years for this template).
		// NOTE: Outer SQL has ORDER BY, so we do NOT need to sort here.
		std::lock_guard<std::mutex> guard(g.lock);

		idx_t out_row = 0;
		const idx_t n = years * 4;

		for (idx_t i = 0; i < n; ++i) {
			if (g.counts[i] == 0) continue;

			const idx_t yy = i / 4;
			const idx_t cs = i - yy * 4;

			const idx_t c_idx = cs / 2;
			const idx_t s_idx = cs - c_idx * 2;

			const int32_t year = (int32_t)(year_lo + (int32_t)yy);

			out.SetValue(0, out_row, Value(g.c_vals[c_idx]));
			out.SetValue(1, out_row, Value(g.s_vals[s_idx]));
			out.SetValue(2, out_row, Value::BIGINT(year));
			out.SetValue(3, out_row, Value::HUGEINT(ToHugeint(g.sums[i])));

			out_row++;
			// In the intended workload (SSB), out_row fits in one vector.
			// If you ever expect a very wide year range, you'd need multi-chunk output support.
			if (out_row >= STANDARD_VECTOR_SIZE) break;
		}

		out.SetCardinality(out_row);
	} else {
		out.SetCardinality(0);
	}

	return OperatorFinalizeResultType::FINISHED;
}

// ------------------------------------------------------------
// Extension registration
// ------------------------------------------------------------

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction f("dbweaver", {LogicalTypeId::TABLE, LogicalType::INTEGER, LogicalType::INTEGER}, nullptr, FnBind,
	                FnInit, FnInitLocal);
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