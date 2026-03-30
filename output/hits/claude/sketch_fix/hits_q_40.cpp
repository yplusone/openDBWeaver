/*
query_template: SELECT TraficSourceID, SearchEngineID, AdvEngineID,
                       CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src,
                       URL AS Dst,
                       COUNT(*) AS PageViews
                FROM hits
                WHERE CounterID = 62
                  AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                  AND IsRefresh = 0
                GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst
                ORDER BY PageViews DESC
                LIMIT 10 OFFSET 1000;

split_template: select * from dbweaver((SELECT TraficSourceID, SearchEngineID, AdvEngineID, Referer, URL
                                       FROM hits
                                       WHERE (CounterID=62) AND (IsRefresh=0)));

query_example: SELECT TraficSourceID, SearchEngineID, AdvEngineID,
                      CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src,
                      URL AS Dst,
                      COUNT(*) AS PageViews
               FROM hits
               WHERE CounterID = 62
                 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                 AND IsRefresh = 0
               GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst
               ORDER BY PageViews DESC
               LIMIT 10 OFFSET 1000;

split_query: select * from dbweaver((SELECT TraficSourceID, SearchEngineID, AdvEngineID, Referer, URL
                                    FROM hits
                                    WHERE (CounterID=62) AND (IsRefresh=0)));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>
#include <algorithm>

namespace duckdb {

// ============================================================
// Constants for this query
// ============================================================

static constexpr idx_t TOPK_LIMIT = 10;
static constexpr idx_t TOPK_OFFSET = 1000;
static constexpr idx_t TOPK_NEED = TOPK_LIMIT + TOPK_OFFSET; // 1010

// ============================================================
// Helpers
// ============================================================

static inline bool IsIntegral(LogicalTypeId id) {
	return id == LogicalTypeId::TINYINT || id == LogicalTypeId::SMALLINT || id == LogicalTypeId::INTEGER ||
	       id == LogicalTypeId::BIGINT || id == LogicalTypeId::UTINYINT || id == LogicalTypeId::USMALLINT ||
	       id == LogicalTypeId::UINTEGER || id == LogicalTypeId::UBIGINT || id == LogicalTypeId::BOOLEAN;
}

static inline int64_t ReadInt64At(const UnifiedVectorFormat &uvf, idx_t ridx, LogicalTypeId tid) {
	const idx_t idx = uvf.sel->get_index(ridx);
	switch (tid) {
	case LogicalTypeId::BOOLEAN:
		return (int64_t)((bool *)uvf.data)[idx];
	case LogicalTypeId::TINYINT:
		return (int64_t)((int8_t *)uvf.data)[idx];
	case LogicalTypeId::SMALLINT:
		return (int64_t)((int16_t *)uvf.data)[idx];
	case LogicalTypeId::INTEGER:
		return (int64_t)((int32_t *)uvf.data)[idx];
	case LogicalTypeId::BIGINT:
		return (int64_t)((int64_t *)uvf.data)[idx];
	case LogicalTypeId::UTINYINT:
		return (int64_t)((uint8_t *)uvf.data)[idx];
	case LogicalTypeId::USMALLINT:
		return (int64_t)((uint16_t *)uvf.data)[idx];
	case LogicalTypeId::UINTEGER:
		return (int64_t)((uint32_t *)uvf.data)[idx];
	case LogicalTypeId::UBIGINT:
		return (int64_t)((uint64_t *)uvf.data)[idx];
	default:
		return 0;
	}
}

static inline bool StringEquals(const string_t &a, const string_t &b) noexcept {
	const auto al = a.GetSize();
	const auto bl = b.GetSize();
	if (al != bl) {
		return false;
	}
	if (al == 0) {
		return true;
	}
	return std::memcmp(a.GetDataUnsafe(), b.GetDataUnsafe(), al) == 0;
}

static inline size_t HashString(const string_t &s) {
	return duckdb::Hash(s);
}

static inline int CmpStringLex(const string_t &a, const string_t &b) {
	const auto al = a.GetSize();
	const auto bl = b.GetSize();
	const auto ml = std::min(al, bl);
	if (ml > 0) {
		const int c = std::memcmp(a.GetDataUnsafe(), b.GetDataUnsafe(), ml);
		if (c != 0) {
			return c;
		}
	}
	if (al == bl) {
		return 0;
	}
	return (al < bl) ? -1 : 1;
}

// ============================================================
// Key: (TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst)
// ============================================================

struct HashedKey {
	int32_t trafic_source_id;
	int32_t search_engine_id;
	int32_t adv_engine_id;
	string_t src;
	string_t dst;
	size_t hash;

	bool operator==(const HashedKey &o) const noexcept {
		if (hash != o.hash) return false;
		if (trafic_source_id != o.trafic_source_id) return false;
		if (search_engine_id != o.search_engine_id) return false;
		if (adv_engine_id != o.adv_engine_id) return false;
		if (!StringEquals(src, o.src)) return false;
		return StringEquals(dst, o.dst);
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept {
		return k.hash;
	}
};

static inline size_t HashCompositeKey(int32_t ts, int32_t se, int32_t adv, const string_t &src, const string_t &dst) {
	size_t h = CombineHash(duckdb::Hash(ts), duckdb::Hash(se));
	h = CombineHash(h, duckdb::Hash(adv));
	h = CombineHash(h, HashString(src));
	h = CombineHash(h, HashString(dst));
	return h;
}

// ============================================================
// Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

// ============================================================
// Global state
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;

	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};
	std::atomic<bool> result_emitted{false};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

// ============================================================
// Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;

	inline void AddOne(int32_t ts, int32_t se, int32_t adv, const string_t &src_in, const string_t &dst_in) {
		const size_t h = HashCompositeKey(ts, se, adv, src_in, dst_in);
		HashedKey probe{ts, se, adv, src_in, dst_in, h};

		auto it = map.find(probe);
		if (it != map.end()) {
			it->second += 1;
			return;
		}

		string_t src = src_in;
		if (src.GetSize() != 0) {
			src = heap.AddString(src_in);
		}
		string_t dst = heap.AddString(dst_in);

		map.emplace(HashedKey{ts, se, adv, src, dst, h}, 1);
	}
};

// ============================================================
// Init / Bind
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return unique_ptr<GlobalTableFunctionState>(make_uniq<FnGlobalState>());
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);
	return unique_ptr<LocalTableFunctionState>(make_uniq<FnLocalState>());
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	return_types.push_back(LogicalType::INTEGER); // TraficSourceID
	return_types.push_back(LogicalType::INTEGER); // SearchEngineID
	return_types.push_back(LogicalType::INTEGER); // AdvEngineID
	return_types.push_back(LogicalType::VARCHAR); // Src
	return_types.push_back(LogicalType::VARCHAR); // Dst
	return_types.push_back(LogicalType::BIGINT);  // PageViews

	names.push_back("TraficSourceID");
	names.push_back("SearchEngineID");
	names.push_back("AdvEngineID");
	names.push_back("Src");
	names.push_back("Dst");
	names.push_back("PageViews");

	return make_uniq<FnBindData>();
}

// ============================================================
// Execute
// Input: TraficSourceID, SearchEngineID, AdvEngineID, Referer, URL
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 5) {
		throw InvalidInputException(
		    "dbweaver expects five columns: TraficSourceID, SearchEngineID, AdvEngineID, Referer, URL");
	}

	auto &v_ts = input.data[0];
	auto &v_se = input.data[1];
	auto &v_adv = input.data[2];
	auto &v_ref = input.data[3];
	auto &v_url = input.data[4];

	const auto t_ts = v_ts.GetType().id();
	const auto t_se = v_se.GetType().id();
	const auto t_adv = v_adv.GetType().id();

	if (!IsIntegral(t_ts) || !IsIntegral(t_se) || !IsIntegral(t_adv)) {
		throw InvalidInputException("dbweaver expects integral types for TraficSourceID/SearchEngineID/AdvEngineID");
	}
	if (v_ref.GetType().id() != LogicalTypeId::VARCHAR || v_url.GetType().id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("dbweaver expects Referer and URL as VARCHAR");
	}

	UnifiedVectorFormat u_ts, u_se, u_adv, u_ref, u_url;
	v_ts.ToUnifiedFormat(input.size(), u_ts);
	v_se.ToUnifiedFormat(input.size(), u_se);
	v_adv.ToUnifiedFormat(input.size(), u_adv);
	v_ref.ToUnifiedFormat(input.size(), u_ref);
	v_url.ToUnifiedFormat(input.size(), u_url);

	auto &val_ts = u_ts.validity;
	auto &val_se = u_se.validity;
	auto &val_adv = u_adv.validity;
	auto &val_ref = u_ref.validity;
	auto &val_url = u_url.validity;

	auto *ref_data = (string_t *)u_ref.data;
	auto *url_data = (string_t *)u_url.data;

	const bool all_valid =
	    val_ts.AllValid() && val_se.AllValid() && val_adv.AllValid() && val_ref.AllValid() && val_url.AllValid();

	constexpr idx_t BATCH = 8;

	if (all_valid) {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min<idx_t>(BATCH, input.size() - rr);

			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;

				const int32_t ts = (int32_t)ReadInt64At(u_ts, ridx, t_ts);
				const int32_t se = (int32_t)ReadInt64At(u_se, ridx, t_se);
				const int32_t adv = (int32_t)ReadInt64At(u_adv, ridx, t_adv);

				const idx_t ref_idx = u_ref.sel->get_index(ridx);
				const idx_t url_idx = u_url.sel->get_index(ridx);

				const string_t referer = ref_data[ref_idx];
				const string_t url = url_data[url_idx];

				const string_t src = (se == 0 && adv == 0) ? referer : string_t();
				l.AddOne(ts, se, adv, src, url);
			}
		}
	} else {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min<idx_t>(BATCH, input.size() - rr);

			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;

				const idx_t ts_i = u_ts.sel->get_index(ridx);
				const idx_t se_i = u_se.sel->get_index(ridx);
				const idx_t adv_i = u_adv.sel->get_index(ridx);
				const idx_t ref_i = u_ref.sel->get_index(ridx);
				const idx_t url_i = u_url.sel->get_index(ridx);

				if (!val_ts.RowIsValid(ts_i)) continue;
				if (!val_se.RowIsValid(se_i)) continue;
				if (!val_adv.RowIsValid(adv_i)) continue;
				if (!val_ref.RowIsValid(ref_i)) continue;
				if (!val_url.RowIsValid(url_i)) continue;

				const int32_t ts = (int32_t)ReadInt64At(u_ts, ridx, t_ts);
				const int32_t se = (int32_t)ReadInt64At(u_se, ridx, t_se);
				const int32_t adv = (int32_t)ReadInt64At(u_adv, ridx, t_adv);

				const string_t referer = ref_data[ref_i];
				const string_t url = url_data[url_i];

				const string_t src = (se == 0 && adv == 0) ? referer : string_t();
				l.AddOne(ts, se, adv, src, url);
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
// Merge helper
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (auto &kv : local.map) {
		const HashedKey &k = kv.first;
		const int64_t cnt = kv.second;
		if (cnt <= 0) {
			continue;
		}

		auto it = g.map.find(k);
		if (it != g.map.end()) {
			it->second += cnt;
			continue;
		}

		string_t g_src = k.src;
		if (g_src.GetSize() != 0) {
			g_src = g.heap.AddString(k.src);
		}
		string_t g_dst = g.heap.AddString(k.dst);

		g.map.emplace(
		    HashedKey{k.trafic_source_id, k.search_engine_id, k.adv_engine_id, g_src, g_dst, k.hash},
		    cnt);
	}
}

// ============================================================
// Final output: LIMIT 10 OFFSET 1000 over ORDER BY PageViews DESC
// For exact pruning, heap comparison must include full tie-break.
// ============================================================

struct TopRow {
	int64_t c;
	int32_t ts;
	int32_t se;
	int32_t adv;
	string_t src;
	string_t dst;
	size_t hash;
};

// Return true if a is better (should appear earlier in final sorted order) than b
static inline bool BetterRow(const TopRow &a, const TopRow &b) {
	if (a.c != b.c) return a.c > b.c;
	if (a.ts != b.ts) return a.ts < b.ts;
	if (a.se != b.se) return a.se < b.se;
	if (a.adv != b.adv) return a.adv < b.adv;
	int c = CmpStringLex(a.src, b.src);
	if (c != 0) return c < 0;
	c = CmpStringLex(a.dst, b.dst);
	if (c != 0) return c < 0;
	return a.hash < b.hash;
}

// priority_queue top = worst row among kept rows
struct TopRowWorstCmp {
	bool operator()(const TopRow &a, const TopRow &b) const {
		return BetterRow(a, b);
	}
};

template <class MapT>
static void EmitOffsetLimitFromMap(MapT &map_ref, DataChunk &out) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowWorstCmp> pq;

	for (auto it = map_ref.begin(); it != map_ref.end(); ++it) {
		const HashedKey &k = it->first;
		const int64_t cnt = it->second;
		if (cnt <= 0) {
			continue;
		}

		TopRow row;
		row.c = cnt;
		row.ts = k.trafic_source_id;
		row.se = k.search_engine_id;
		row.adv = k.adv_engine_id;
		row.src = k.src;
		row.dst = k.dst;
		row.hash = k.hash;

		if (pq.size() < TOPK_NEED) {
			pq.push(row);
		} else if (BetterRow(row, pq.top())) {
			pq.pop();
			pq.push(row);
		}
	}

	std::vector<TopRow> top;
	top.reserve(pq.size());
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}

	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		return BetterRow(a, b);
	});

	if (top.size() <= TOPK_OFFSET) {
		out.SetCardinality(0);
		return;
	}

	auto *out_ts = FlatVector::GetData<int32_t>(out.data[0]);
	auto *out_se = FlatVector::GetData<int32_t>(out.data[1]);
	auto *out_adv = FlatVector::GetData<int32_t>(out.data[2]);
	auto *out_src = FlatVector::GetData<string_t>(out.data[3]);
	auto *out_dst = FlatVector::GetData<string_t>(out.data[4]);
	auto *out_pv = FlatVector::GetData<int64_t>(out.data[5]);

	const idx_t start = TOPK_OFFSET;
	const idx_t end = std::min<idx_t>((idx_t)top.size(), TOPK_OFFSET + TOPK_LIMIT);

	idx_t out_idx = 0;
	for (idx_t i = start; i < end; ++i) {
		out_ts[out_idx] = top[i].ts;
		out_se[out_idx] = top[i].se;
		out_adv[out_idx] = top[i].adv;
		out_src[out_idx] = StringVector::AddString(out.data[3], top[i].src);
		out_dst[out_idx] = StringVector::AddString(out.data[4], top[i].dst);
		out_pv[out_idx] = top[i].c;
		++out_idx;
	}
	out.SetCardinality(out_idx);
}

// ============================================================
// Finalize
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	if (!l.merged) {
		std::lock_guard<std::mutex> guard(g.lock);
		MergeLocalIntoGlobal(l, g);
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_acq_rel);
	}

	const idx_t merged = g.merged_local_states.load(std::memory_order_acquire);
	const idx_t active = g.active_local_states.load(std::memory_order_acquire);

	if (active > 0 && merged == active) {
		bool expected = false;
		if (g.result_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> guard(g.lock);
			EmitOffsetLimitFromMap(g.map, out);
			return OperatorFinalizeResultType::FINISHED;
		}
	}

	out.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
// Extension load
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