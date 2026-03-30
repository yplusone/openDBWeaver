/*
query_template: SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews
                FROM hits
                WHERE CounterID = 62
                  AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                  AND IsRefresh = 0
                  AND DontCountHits = 0
                  AND URLHash = 2868770270353813622
                GROUP BY WindowClientWidth, WindowClientHeight
                ORDER BY PageViews DESC
                LIMIT 10 OFFSET 10000;

split_template: select * from dbweaver((SELECT WindowClientWidth, WindowClientHeight
                                       FROM hits
                                       WHERE (CounterID=62)
                                         AND (IsRefresh=0)
                                         AND (DontCountHits=0)
                                         AND (URLHash=2868770270353813622)));

query_example: SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews
               FROM hits
               WHERE CounterID = 62
                 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                 AND IsRefresh = 0
                 AND DontCountHits = 0
                 AND URLHash = 2868770270353813622
               GROUP BY WindowClientWidth, WindowClientHeight
               ORDER BY PageViews DESC
               LIMIT 10 OFFSET 10000;

split_query: select * from dbweaver((SELECT WindowClientWidth, WindowClientHeight
                                    FROM hits
                                    WHERE (CounterID=62)
                                      AND (IsRefresh=0)
                                      AND (DontCountHits=0)
                                      AND (URLHash=2868770270353813622)));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
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

static constexpr idx_t TOPK_LIMIT  = 10;
static constexpr idx_t TOPK_OFFSET = 10000;
static constexpr idx_t TOPK_NEED   = TOPK_LIMIT + TOPK_OFFSET; // 10010

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

static inline uint64_t Pack(int32_t w, int32_t h) {
	return (static_cast<uint64_t>(static_cast<uint32_t>(w)) << 32) | static_cast<uint32_t>(h);
}

static inline int32_t UnpackW(uint64_t k) { return static_cast<int32_t>(k >> 32); }
static inline int32_t UnpackH(uint64_t k) { return static_cast<int32_t>(k & 0xFFFFFFFF); }

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

	absl::flat_hash_map<uint64_t, int64_t> map;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};
	std::atomic<bool>  result_emitted{false};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

// ============================================================
// Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	absl::flat_hash_map<uint64_t, int64_t> map;

	inline void AddOne(uint64_t key) {
		map[key]++;
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
	return_types.push_back(LogicalType::INTEGER); // WindowClientWidth
	return_types.push_back(LogicalType::INTEGER); // WindowClientHeight
	return_types.push_back(LogicalType::BIGINT);  // PageViews

	names.push_back("WindowClientWidth");
	names.push_back("WindowClientHeight");
	names.push_back("PageViews");

	return make_uniq<FnBindData>();
}

// ============================================================
// Execute (expects pre-filtered input: WindowClientWidth, WindowClientHeight)
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 2) {
		throw InvalidInputException("dbweaver expects two columns: WindowClientWidth, WindowClientHeight");
	}

	auto &v_w = input.data[0];
	auto &v_h = input.data[1];

	const auto t_w = v_w.GetType().id();
	const auto t_h = v_h.GetType().id();

	if (!IsIntegral(t_w) || !IsIntegral(t_h)) {
		throw InvalidInputException("dbweaver expects WindowClientWidth/WindowClientHeight as integral types");
	}
	// Fast path: Specialize for common FlatVector and INTEGER (int32_t) types
	if (v_w.GetVectorType() == VectorType::FLAT_VECTOR && v_h.GetVectorType() == VectorType::FLAT_VECTOR &&
	    t_w == LogicalTypeId::INTEGER && t_h == LogicalTypeId::INTEGER) {
		auto data_w = FlatVector::GetData<int32_t>(v_w);
		auto data_h = FlatVector::GetData<int32_t>(v_h);
		auto &mask_w = FlatVector::Validity(v_w);
		auto &mask_h = FlatVector::Validity(v_h);
		const idx_t size = input.size();

		if (mask_w.AllValid() && mask_h.AllValid()) {
			for (idx_t rr = 0; rr < size; rr += 8) {
				const idx_t n = std::min<idx_t>(8, size - rr);
				uint64_t keys[8];
				for (idx_t k = 0; k < n; k++) {
					keys[k] = Pack(data_w[rr + k], data_h[rr + k]);
				}
				for (idx_t k = 0; k < n; k++) {
					l.AddOne(keys[k]);
				}
			}
		} else {
			for (idx_t i = 0; i < size; i++) {
				if (mask_w.RowIsValid(i) && mask_h.RowIsValid(i)) {
					l.AddOne(Pack(data_w[i], data_h[i]));
				}
			}
		}
		return OperatorResultType::NEED_MORE_INPUT;
	}

	UnifiedVectorFormat u_w, u_h;
	v_w.ToUnifiedFormat(input.size(), u_w);
	v_h.ToUnifiedFormat(input.size(), u_h);

	auto &val_w = u_w.validity;
	auto &val_h = u_h.validity;

	const bool all_valid = val_w.AllValid() && val_h.AllValid();

	constexpr idx_t BATCH = 8;

	if (all_valid) {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min<idx_t>(BATCH, input.size() - rr);
			uint64_t keys[BATCH];
			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;
				keys[k] = Pack((int32_t)ReadInt64At(u_w, ridx, t_w), (int32_t)ReadInt64At(u_h, ridx, t_h));
			}
			for (idx_t k = 0; k < n; ++k) {
				l.AddOne(keys[k]);
			}
		}
	} else {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min<idx_t>(BATCH, input.size() - rr);
			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;
				const idx_t wi = u_w.sel->get_index(ridx);
				const idx_t hi = u_h.sel->get_index(ridx);

				if (!val_w.RowIsValid(wi)) continue;
				if (!val_h.RowIsValid(hi)) continue;

				l.AddOne(Pack((int32_t)ReadInt64At(u_w, ridx, t_w), (int32_t)ReadInt64At(u_h, ridx, t_h)));
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
// Merge helper
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (auto const& kv : local.map) {
		if (kv.second <= 0) continue;
		g.map[kv.first] += kv.second;
	}
}

// ============================================================
// Finalize: OFFSET+LIMIT TopK (need top 10010 by PageViews)
// ============================================================

struct TopRow {
	int64_t c;
	int32_t w;
	int32_t h;
};

// "a is better than b" (earlier in final ORDER BY)
static inline bool BetterRow(const TopRow &a, const TopRow &b) {
	if (a.c != b.c) return a.c > b.c;
	if (a.w != b.w) return a.w < b.w;
	return a.h < b.h;
}

// priority_queue top is the WORST row among kept rows
struct WorstCmp {
	bool operator()(const TopRow &a, const TopRow &b) const {
		return BetterRow(a, b);
	}
};

template <class MapT>
static void EmitOffsetLimitFromMap(MapT &map_ref, DataChunk &out) {
	std::priority_queue<TopRow, std::vector<TopRow>, WorstCmp> pq;

	for (auto const& kv : map_ref) {
		if (kv.second <= 0) continue;

		TopRow row;
		row.c = kv.second;
		row.w = UnpackW(kv.first);
		row.h = UnpackH(kv.first);

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

	auto *out_w  = FlatVector::GetData<int32_t>(out.data[0]);
	auto *out_h  = FlatVector::GetData<int32_t>(out.data[1]);
	auto *out_pv = FlatVector::GetData<int64_t>(out.data[2]);

	const idx_t start = TOPK_OFFSET;
	const idx_t end   = std::min<idx_t>((idx_t)top.size(), TOPK_OFFSET + TOPK_LIMIT);

	idx_t out_idx = 0;
	for (idx_t i = start; i < end; ++i) {
		out_w[out_idx]  = top[i].w;
		out_h[out_idx]  = top[i].h;
		out_pv[out_idx] = top[i].c;
		++out_idx;
	}
	out.SetCardinality(out_idx);
}

// ============================================================
// Finalize: stable "merge then single emitter" pattern
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	// merge once
	if (!l.merged) {
		std::lock_guard<std::mutex> guard(g.lock);
		MergeLocalIntoGlobal(l, g);
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_acq_rel);
	}

	const idx_t merged = g.merged_local_states.load(std::memory_order_acquire);
	const idx_t active = g.active_local_states.load(std::memory_order_acquire);

	// last merger emits exactly once
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
