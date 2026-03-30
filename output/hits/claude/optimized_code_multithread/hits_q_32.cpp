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
	size_t operator()(const HashedKey &k) const noexcept {
		return k.hash;
	}
};

static inline size_t HashCompositeKey(int64_t watch_id, int64_t client_ip) {
	return CombineHash(duckdb::Hash(watch_id), duckdb::Hash(client_ip));
}

// ============================================================
//  Agg payload (Inlined in Map Value)
// ============================================================

struct AggPayload {
	int64_t c;
	int64_t sum_isrefresh;
	int64_t sum_resolutionwidth;

uint64_t first_seen_seq;
};

// ============================================================
//  Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<FnBindData>();
	}
	bool Equals(const FunctionData &) const override {
		return true;
	}
};

// ============================================================
//  Global state with Partitioned Hash Maps
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	static constexpr idx_t NUM_PARTITIONS = 64;

	struct Partition {
		std::mutex lock;
		absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> map;
	};

	std::unique_ptr<Partition> partitions[NUM_PARTITIONS];

	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};
	std::atomic<bool> output_emitted {false};

	FnGlobalState() {
		for (idx_t i = 0; i < NUM_PARTITIONS; i++) {
			partitions[i] = make_uniq<Partition>();
		}
	}

	idx_t MaxThreads() const override {
		return std::numeric_limits<idx_t>::max();
	}
};

// ============================================================
//  Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	uint64_t local_seq = 0;

	absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> map;

	inline AggPayload *FindOrInsertPayload(int64_t watch_id, int64_t client_ip, size_t hash) {
		HashedKey tmp {watch_id, client_ip, hash};
		auto res = map.try_emplace(tmp);
		AggPayload &p = res.first->second;
		if (res.second) {
			p.c = 0;
			p.sum_isrefresh = 0;
			p.sum_resolutionwidth = 0;
			p.first_seen_seq = local_seq++;
		}
		return &p;
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
//  Helpers for integral reads
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
// ============================================================
//  Specialized Processing
// ============================================================

template <typename T_WATCH, typename T_IP, typename T_REFRESH, typename T_WIDTH>
static void FnExecuteSpecialized(FnLocalState &l, idx_t count, bool all_valid, const UnifiedVectorFormat &u_watch,
                                 const UnifiedVectorFormat &u_ip, const UnifiedVectorFormat &u_refresh,
                                 const UnifiedVectorFormat &u_width) {
	const T_WATCH *w_ptr = (const T_WATCH *)u_watch.data;
	const T_IP *i_ptr = (const T_IP *)u_ip.data;
	const T_REFRESH *r_ptr = (const T_REFRESH *)u_refresh.data;
	const T_WIDTH *d_ptr = (const T_WIDTH *)u_width.data;

	int64_t watch_ids[STANDARD_VECTOR_SIZE];
	int64_t client_ips[STANDARD_VECTOR_SIZE];
	int64_t refresh_vals[STANDARD_VECTOR_SIZE];
	int64_t width_vals[STANDARD_VECTOR_SIZE];
	size_t hashes[STANDARD_VECTOR_SIZE];
	bool valid_mask[STANDARD_VECTOR_SIZE];

	for (idx_t i = 0; i < count; i++) {
		const idx_t wi = u_watch.sel->get_index(i);
		const idx_t ii = u_ip.sel->get_index(i);
		const idx_t ri = u_refresh.sel->get_index(i);
		const idx_t di = u_width.sel->get_index(i);

		bool row_valid = all_valid || (u_watch.validity.RowIsValid(wi) && u_ip.validity.RowIsValid(ii) &&
		                               u_refresh.validity.RowIsValid(ri) && u_width.validity.RowIsValid(di));
		valid_mask[i] = row_valid;
		if (row_valid) {
			watch_ids[i] = (int64_t)w_ptr[wi];
			client_ips[i] = (int64_t)i_ptr[ii];
			refresh_vals[i] = (int64_t)r_ptr[ri];
			width_vals[i] = (int64_t)d_ptr[di];
			hashes[i] = HashCompositeKey(watch_ids[i], client_ips[i]);
		}
	}

	constexpr idx_t PREFETCH_DIST = 16;
	for (idx_t i = 0; i < count; i++) {
		if (!valid_mask[i])
			continue;

		if (i + PREFETCH_DIST < count && valid_mask[i + PREFETCH_DIST]) {
			__builtin_prefetch(&l.map, 0, 1);
		}

		AggPayload *p = l.FindOrInsertPayload(watch_ids[i], client_ips[i], hashes[i]);
		p->c += 1;
		p->sum_isrefresh += refresh_vals[i];
		p->sum_resolutionwidth += width_vals[i];
	}
}

static void FnExecuteGeneric(FnLocalState &l, idx_t count, bool all_valid, const UnifiedVectorFormat &u_watch,
                             const UnifiedVectorFormat &u_ip, const UnifiedVectorFormat &u_refresh,
                             const UnifiedVectorFormat &u_width, LogicalTypeId t_watch, LogicalTypeId t_ip,
                             LogicalTypeId t_refresh, LogicalTypeId t_width) {
	int64_t watch_ids[STANDARD_VECTOR_SIZE];
	int64_t client_ips[STANDARD_VECTOR_SIZE];
	int64_t refresh_vals[STANDARD_VECTOR_SIZE];
	int64_t width_vals[STANDARD_VECTOR_SIZE];
	size_t hashes[STANDARD_VECTOR_SIZE];
	bool valid_mask[STANDARD_VECTOR_SIZE];

	for (idx_t i = 0; i < count; i++) {
		bool row_valid = all_valid || (u_watch.validity.RowIsValid(u_watch.sel->get_index(i)) &&
		                               u_ip.validity.RowIsValid(u_ip.sel->get_index(i)) &&
		                               u_refresh.validity.RowIsValid(u_refresh.sel->get_index(i)) &&
		                               u_width.validity.RowIsValid(u_width.sel->get_index(i)));
		valid_mask[i] = row_valid;
		if (row_valid) {
			watch_ids[i] = ReadInt64At(u_watch, i, t_watch);
			client_ips[i] = ReadInt64At(u_ip, i, t_ip);
			refresh_vals[i] = ReadInt64At(u_refresh, i, t_refresh);
			width_vals[i] = ReadInt64At(u_width, i, t_width);
			hashes[i] = HashCompositeKey(watch_ids[i], client_ips[i]);
		}
	}

	constexpr idx_t PREFETCH_DIST = 16;
	for (idx_t i = 0; i < count; i++) {
		if (!valid_mask[i])
			continue;

		if (i + PREFETCH_DIST < count && valid_mask[i + PREFETCH_DIST]) {
			__builtin_prefetch(&l.map, 0, 1);
		}

		AggPayload *p = l.FindOrInsertPayload(watch_ids[i], client_ips[i], hashes[i]);
		p->c += 1;
		p->sum_isrefresh += refresh_vals[i];
		p->sum_resolutionwidth += width_vals[i];
	}
}

// ============================================================
//  Execute
// ============================================================


static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0)
		return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();

	// Expect 4 columns: WatchID, ClientIP, IsRefresh, ResolutionWidth
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
		throw InvalidInputException(
		    "dbweaver expects integral types for WatchID, ClientIP, IsRefresh, ResolutionWidth");
	}

	UnifiedVectorFormat u_watch, u_ip, u_refresh, u_width;
	v_watch.ToUnifiedFormat(input.size(), u_watch);
	v_ip.ToUnifiedFormat(input.size(), u_ip);
	v_refresh.ToUnifiedFormat(input.size(), u_refresh);
	v_width.ToUnifiedFormat(input.size(), u_width);

	const bool all_valid =
	    u_watch.validity.AllValid() && u_ip.validity.AllValid() && 
	    u_refresh.validity.AllValid() && u_width.validity.AllValid();
	const idx_t count = input.size();

	if (t_watch == LogicalTypeId::BIGINT && t_ip == LogicalTypeId::INTEGER && t_refresh == LogicalTypeId::SMALLINT &&
	    t_width == LogicalTypeId::SMALLINT) {
		FnExecuteSpecialized<int64_t, int32_t, int16_t, int16_t>(l, count, all_valid, u_watch, u_ip, u_refresh, u_width);
	} else if (t_watch == LogicalTypeId::BIGINT && t_ip == LogicalTypeId::BIGINT && t_refresh == LogicalTypeId::BIGINT &&
	           t_width == LogicalTypeId::BIGINT) {
		FnExecuteSpecialized<int64_t, int64_t, int64_t, int64_t>(l, count, all_valid, u_watch, u_ip, u_refresh, u_width);
	} else {
		FnExecuteGeneric(l, count, all_valid, u_watch, u_ip, u_refresh, u_width, t_watch, t_ip, t_refresh, t_width);
	}


	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize Helpers: Top-K aggregation
// ============================================================

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
		if (a.c != b.c)
			return a.c > b.c;
		return a.first_seen_seq < b.first_seen_seq;
	}
};

static void CollectTop10(std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq,
                         const absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> &map) {
	for (auto const &kv : map) {
		const HashedKey &k = kv.first;
		const AggPayload &p = kv.second;
		if (p.c <= 0)
			continue;
		TopRow row {p.c, k.watch_id, k.client_ip, p.sum_isrefresh, p.sum_resolutionwidth, p.first_seen_seq};
		if (pq.size() < 10) {
			pq.push(row);
		} else {
			const auto &worst = pq.top();
			if (row.c > worst.c || (row.c == worst.c && row.first_seen_seq < worst.first_seen_seq)) {
				pq.pop();
				pq.push(row);
			}
		}
	}
}

static void EmitFinalChunk(DataChunk &out, std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq) {
	std::vector<TopRow> top;
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}

	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.c != b.c)
			return a.c > b.c;
		if (a.first_seen_seq != b.first_seen_seq)
			return a.first_seen_seq < b.first_seen_seq;
		if (a.client_ip != b.client_ip)
			return a.client_ip < b.client_ip;
		return a.watch_id < b.watch_id;
	});

	auto *out_ip = FlatVector::GetData<int64_t>(out.data[0]);
	auto *out_watch = FlatVector::GetData<int64_t>(out.data[1]);
	auto *out_avgw = FlatVector::GetData<double>(out.data[2]);
	auto *out_c = FlatVector::GetData<int64_t>(out.data[3]);
	auto *out_sumref = FlatVector::GetData<int64_t>(out.data[4]);

	idx_t out_idx = 0;
	for (auto &r : top) {
		out_ip[out_idx] = r.client_ip;
		out_watch[out_idx] = r.watch_id;
		out_avgw[out_idx] = (double)r.sum_resolutionwidth / (double)r.c;
		out_c[out_idx] = r.c;
		out_sumref[out_idx] = r.sum_isrefresh;
		++out_idx;
	}
	out.SetCardinality(out_idx);
}

// ============================================================
//  Finalize: Parallel Partitioned Merge
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	// Single-thread fast path
	if (active == 1) {
		bool expected = false;
		if (!g.output_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			out.SetCardinality(0);
			return OperatorFinalizeResultType::FINISHED;
		}
		std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
		CollectTop10(pq, l.map);
		EmitFinalChunk(out, pq);
		return OperatorFinalizeResultType::FINISHED;
	}

	// Parallel merge into partitions
	if (!l.merged) {
		std::vector<const std::pair<const HashedKey, AggPayload> *> bins[FnGlobalState::NUM_PARTITIONS];
		for (auto const &kv : l.map) {
			bins[kv.first.hash % FnGlobalState::NUM_PARTITIONS].push_back(&kv);
		}

		for (idx_t i = 0; i < FnGlobalState::NUM_PARTITIONS; i++) {
			if (bins[i].empty())
				continue;
			auto &part = *g.partitions[i];
			std::lock_guard<std::mutex> guard(part.lock);
			for (auto const *ptr : bins[i]) {
				const auto &k = ptr->first;
				const auto &lp = ptr->second;
				auto res = part.map.try_emplace(k, lp);
				if (!res.second) {
					auto &gp = res.first->second;
					gp.c += lp.c;
					gp.sum_isrefresh += lp.sum_isrefresh;
					gp.sum_resolutionwidth += lp.sum_resolutionwidth;
					if (lp.first_seen_seq < gp.first_seen_seq) {
						gp.first_seen_seq = lp.first_seen_seq;
					}
				}
			}
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	if (merged == active) {
		bool expected = false;
		if (g.output_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
			for (idx_t i = 0; i < FnGlobalState::NUM_PARTITIONS; i++) {
				CollectTop10(pq, g.partitions[i]->map);
			}
			EmitFinalChunk(out, pq);
			return OperatorFinalizeResultType::FINISHED;
	}
	}

	out.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
//  Extension load
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
