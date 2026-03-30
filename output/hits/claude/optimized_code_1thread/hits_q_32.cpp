#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/function/table_function.hpp"

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>
#include <algorithm>

namespace duckdb {

// ============================================================
//  Key: WatchID + ClientIP + precomputed hash
// ============================================================

struct HashedKey {
	int64_t watch_id;
	int64_t client_ip;
	size_t hash;

	bool operator==(const HashedKey &o) const noexcept {
		return hash == o.hash && watch_id == o.watch_id && client_ip == o.client_ip;
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept { return k.hash; }
};

static inline size_t HashCompositeKey(int64_t watch_id, int64_t client_ip) {
	return CombineHash(duckdb::Hash(watch_id), duckdb::Hash(client_ip));
}

// ============================================================
//  Agg payload + arena (stable addresses)
// ============================================================

struct AggPayload {
	int64_t c;
	int64_t sum_isrefresh;
	int64_t sum_resolutionwidth;

uint64_t first_seen_seq;
};

struct AggArena {
	static constexpr idx_t BLOCK_SIZE = 4096;

	std::vector<std::unique_ptr<AggPayload[]>> blocks;
	idx_t offset = BLOCK_SIZE;

	inline AggPayload *AllocateZeroed() {
		if (offset >= BLOCK_SIZE) {
			blocks.emplace_back(std::unique_ptr<AggPayload[]>(new AggPayload[BLOCK_SIZE]));
			offset = 0;
		}
		AggPayload *p = &blocks.back()[offset++];
		p->c = 0;
		p->sum_isrefresh = 0;
		p->sum_resolutionwidth = 0;
		p->first_seen_seq = 0;
		return p;
	}
};

// ============================================================
//  States
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;
	AggArena arena;
	absl::flat_hash_map<HashedKey, AggPayload *, HashedKeyHash> map;
	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};
	std::atomic<uint8_t> adopt_stage {0};
	std::atomic<bool> output_emitted {false};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	uint64_t local_seq = 0;
	AggArena arena;
	absl::flat_hash_map<HashedKey, AggPayload *, HashedKeyHash> map;

	inline AggPayload *FindOrInsertPayload(int64_t watch_id, int64_t client_ip, size_t hash) {
		HashedKey tmp {watch_id, client_ip, hash};
		auto it = map.find(tmp);
		if (it != map.end()) return it->second;
		AggPayload *p = arena.AllocateZeroed();
		p->first_seen_seq = local_seq++;
		map.emplace(tmp, p);
		return p;
	}
};

// ============================================================
//  Init / Bind
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
                                       vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::BIGINT); // ClientIP
	return_types.push_back(LogicalType::BIGINT); // WatchID
	return_types.push_back(LogicalType::DOUBLE); // avg_resolutionwidth
	return_types.push_back(LogicalType::BIGINT); // c
	return_types.push_back(LogicalType::BIGINT); // sum_isrefresh

	names.push_back("ClientIP");
	names.push_back("WatchID");
	names.push_back("avg_resolutionwidth");
	names.push_back("c");
	names.push_back("sum_isrefresh");

	return make_uniq<FnBindData>();
}

// ============================================================
//  Materialization Helper (Resolves types once per chunk)
// ============================================================

static inline bool IsIntegral(LogicalTypeId id) {
	return id == LogicalTypeId::TINYINT || id == LogicalTypeId::SMALLINT || id == LogicalTypeId::INTEGER ||
	       id == LogicalTypeId::BIGINT || id == LogicalTypeId::UTINYINT || id == LogicalTypeId::USMALLINT ||
	       id == LogicalTypeId::UINTEGER || id == LogicalTypeId::UBIGINT || id == LogicalTypeId::BOOLEAN;
}

static void FillBuffer(int64_t *dest, const UnifiedVectorFormat &uv, LogicalTypeId tid, idx_t size) {
	const void *data = uv.data;
	const SelectionVector &sel = *uv.sel;

	switch (tid) {
	case LogicalTypeId::BOOLEAN:
		for (idx_t i = 0; i < size; i++) dest[i] = (int64_t)((bool *)data)[sel.get_index(i)];
		break;
	case LogicalTypeId::TINYINT:
		for (idx_t i = 0; i < size; i++) dest[i] = (int64_t)((int8_t *)data)[sel.get_index(i)];
		break;
	case LogicalTypeId::SMALLINT:
		for (idx_t i = 0; i < size; i++) dest[i] = (int64_t)((int16_t *)data)[sel.get_index(i)];
		break;
	case LogicalTypeId::INTEGER:
		for (idx_t i = 0; i < size; i++) dest[i] = (int64_t)((int32_t *)data)[sel.get_index(i)];
		break;
	case LogicalTypeId::BIGINT:
		for (idx_t i = 0; i < size; i++) dest[i] = (int64_t)((int64_t *)data)[sel.get_index(i)];
		break;
	case LogicalTypeId::UTINYINT:
		for (idx_t i = 0; i < size; i++) dest[i] = (int64_t)((uint8_t *)data)[sel.get_index(i)];
		break;
	case LogicalTypeId::USMALLINT:
		for (idx_t i = 0; i < size; i++) dest[i] = (int64_t)((uint16_t *)data)[sel.get_index(i)];
		break;
	case LogicalTypeId::UINTEGER:
		for (idx_t i = 0; i < size; i++) dest[i] = (int64_t)((uint32_t *)data)[sel.get_index(i)];
		break;
	case LogicalTypeId::UBIGINT:
		for (idx_t i = 0; i < size; i++) dest[i] = (int64_t)((uint64_t *)data)[sel.get_index(i)];
		break;
	default:
		for (idx_t i = 0; i < size; i++) dest[i] = 0;
		break;
	}
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;
	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 4) {
		throw InvalidInputException("dbweaver expects four columns: WatchID, ClientIP, IsRefresh, ResolutionWidth");
	}

	auto &v_watch = input.data[0];
	auto &v_ip = input.data[1];
	auto &v_refresh = input.data[2];
	auto &v_width = input.data[3];

	const auto t_watch = v_watch.GetType().id();
	const auto t_ip = v_ip.GetType().id();
	const auto t_refresh = v_refresh.GetType().id();
	const auto t_width = v_width.GetType().id();

	if (!IsIntegral(t_watch) || !IsIntegral(t_ip) || !IsIntegral(t_refresh) || !IsIntegral(t_width)) {
		throw InvalidInputException("dbweaver expects integral types");
	}

	UnifiedVectorFormat u_watch, u_ip, u_refresh, u_width;
	v_watch.ToUnifiedFormat(input.size(), u_watch);
	v_ip.ToUnifiedFormat(input.size(), u_ip);
	v_refresh.ToUnifiedFormat(input.size(), u_refresh);
	v_width.ToUnifiedFormat(input.size(), u_width);

	// Use stack buffers to materialize columns to int64, eliminating the per-row switch.
	// Standard vector size in DuckDB is usually 1024 or 2048.
	int64_t watch_buf[2048];
	int64_t ip_buf[2048];
	int64_t refresh_buf[2048];
	int64_t width_buf[2048];
	if (input.size() > 2048) {
		throw InternalException("Input chunk size exceeds stack buffer capacity");
	}

	FillBuffer(watch_buf, u_watch, t_watch, input.size());
	FillBuffer(ip_buf, u_ip, t_ip, input.size());
	FillBuffer(refresh_buf, u_refresh, t_refresh, input.size());
	FillBuffer(width_buf, u_width, t_width, input.size());

	size_t hash_buf[2048];
	for (idx_t i = 0; i < input.size(); i++) {
		hash_buf[i] = HashCompositeKey(watch_buf[i], ip_buf[i]);
	}

	const bool all_valid =
	    u_watch.validity.AllValid() && u_ip.validity.AllValid() && u_refresh.validity.AllValid() && u_width.validity.AllValid();

	const idx_t D = 16;
	if (all_valid) {
		for (idx_t i = 0; i < input.size(); i++) {
			if (i + D < input.size()) {
				HashedKey pk {watch_buf[i + D], ip_buf[i + D], hash_buf[i + D]};
				l.map.prefetch(pk);
			}
			int64_t w = watch_buf[i];
			int64_t ip = ip_buf[i];
			AggPayload *p = l.FindOrInsertPayload(w, ip, hash_buf[i]);
			p->c += 1;
			p->sum_isrefresh += refresh_buf[i];
			p->sum_resolutionwidth += width_buf[i];
		}
	} else {
		for (idx_t i = 0; i < input.size(); i++) {
			if (!u_watch.validity.RowIsValid(u_watch.sel->get_index(i))) continue;
			if (!u_ip.validity.RowIsValid(u_ip.sel->get_index(i))) continue;
			if (!u_refresh.validity.RowIsValid(u_refresh.sel->get_index(i))) continue;
			if (!u_width.validity.RowIsValid(u_width.sel->get_index(i))) continue;

			if (i + D < input.size()) {
				HashedKey pk {watch_buf[i + D], ip_buf[i + D], hash_buf[i + D]};
				l.map.prefetch(pk);
			}
			int64_t w = watch_buf[i];
			int64_t ip = ip_buf[i];
			AggPayload *p = l.FindOrInsertPayload(w, ip, hash_buf[i]);
			p->c += 1;
			p->sum_isrefresh += refresh_buf[i];
			p->sum_resolutionwidth += width_buf[i];
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (auto &kv : local.map) {
		const HashedKey &k = kv.first;
		AggPayload *lp = kv.second;
		if (!lp || lp->c == 0) continue;
		auto it = g.map.find(k);
		if (it != g.map.end()) {
			AggPayload *gp = it->second;
			gp->c += lp->c;
			gp->sum_isrefresh += lp->sum_isrefresh;
			gp->sum_resolutionwidth += lp->sum_resolutionwidth;
			if (lp->first_seen_seq < gp->first_seen_seq) gp->first_seen_seq = lp->first_seen_seq;
			continue;
		}
		AggPayload *gp = g.arena.AllocateZeroed();
		*gp = *lp;
		g.map.emplace(k, gp);
	}
}

struct TopRow {
	int64_t c;
	int64_t watch_id;
	int64_t client_ip;
	int64_t sum_isrefresh;
	int64_t sum_resolutionwidth;
	uint64_t first_seen_seq;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const {
		if (a.c != b.c) return a.c > b.c;
		return a.first_seen_seq < b.first_seen_seq;
	}
};

static void EmitTop10(DataChunk &out, absl::flat_hash_map<HashedKey, AggPayload *, HashedKeyHash> &map_ref) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	for (auto &kv : map_ref) {
		const HashedKey &k = kv.first;
		AggPayload *p = kv.second;
		if (!p || p->c <= 0) continue;
		TopRow row {p->c, k.watch_id, k.client_ip, p->sum_isrefresh, p->sum_resolutionwidth, p->first_seen_seq};
		if (pq.size() < 10) pq.push(row);
		else {
			const auto &worst = pq.top();
			if (row.c > worst.c || (row.c == worst.c && row.first_seen_seq < worst.first_seen_seq)) {
				pq.pop();
				pq.push(row);
			}
		}
	}
	std::vector<TopRow> top;
	while (!pq.empty()) { top.push_back(pq.top()); pq.pop(); }
	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.c != b.c) return a.c > b.c;
		if (a.first_seen_seq != b.first_seen_seq) return a.first_seen_seq < b.first_seen_seq;
		if (a.client_ip != b.client_ip) return a.client_ip < b.client_ip;
		return a.watch_id < b.watch_id;
	});
	idx_t out_idx = 0;
	for (auto &r : top) {
		FlatVector::GetData<int64_t>(out.data[0])[out_idx] = r.client_ip;
		FlatVector::GetData<int64_t>(out.data[1])[out_idx] = r.watch_id;
		FlatVector::GetData<double>(out.data[2])[out_idx] = (r.c > 0) ? (double)r.sum_resolutionwidth / (double)r.c : 0.0;
		FlatVector::GetData<int64_t>(out.data[3])[out_idx] = r.c;
		FlatVector::GetData<int64_t>(out.data[4])[out_idx] = r.sum_isrefresh;
		++out_idx;
	}
	out.SetCardinality(out_idx);
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();
	const auto active = g.active_local_states.load(std::memory_order_relaxed);
	if (active == 1) {
		bool expected = false;
		if (g.output_emitted.compare_exchange_strong(expected, true)) EmitTop10(out, l.map);
		else out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}
	if (!l.merged) {
		uint8_t expected = 0;
		if (g.adopt_stage.compare_exchange_strong(expected, 1)) {
			std::lock_guard<std::mutex> guard(g.lock);
			if (g.map.empty()) { g.arena = std::move(l.arena); g.map = std::move(l.map); }
			else MergeLocalIntoGlobal(l, g);
			g.adopt_stage.store(2, std::memory_order_release);
		} else {
			while (g.adopt_stage.load(std::memory_order_acquire) == 1) std::this_thread::yield();
			std::lock_guard<std::mutex> guard(g.lock);
			MergeLocalIntoGlobal(l, g);
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}
	if (active > 0 && g.merged_local_states.load(std::memory_order_relaxed) == active) {
		bool expected = false;
		if (g.output_emitted.compare_exchange_strong(expected, true)) EmitTop10(out, g.map);
		else out.SetCardinality(0);
	}
	else out.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

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
