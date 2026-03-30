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

// Partition count must be a power of two
static constexpr idx_t PARTITION_COUNT = 128;

// ============================================================
// Helpers
// ============================================================

static inline bool IsIntegral(LogicalTypeId id) {
	return id == LogicalTypeId::TINYINT || id == LogicalTypeId::SMALLINT || id == LogicalTypeId::INTEGER ||
	       id == LogicalTypeId::BIGINT || id == LogicalTypeId::UTINYINT || id == LogicalTypeId::USMALLINT ||
	       id == LogicalTypeId::UINTEGER || id == LogicalTypeId::UBIGINT || id == LogicalTypeId::BOOLEAN;
}

struct IntColumnReader {
	const void *data;
	const SelectionVector *sel;
	int64_t (*read_func)(const void *, idx_t);

	template <typename T>
	static int64_t ReadT(const void *data, idx_t idx) {
		return (int64_t)((const T *)data)[idx];
	}

	static int64_t ReadBool(const void *data, idx_t idx) {
		return (int64_t)((const bool *)data)[idx];
	}

	void Initialize(LogicalTypeId tid, const UnifiedVectorFormat &uvf) {
		data = uvf.data;
		sel = uvf.sel;
		switch (tid) {
		case LogicalTypeId::BOOLEAN:
			read_func = ReadBool;
			break;
		case LogicalTypeId::TINYINT:
			read_func = ReadT<int8_t>;
			break;
		case LogicalTypeId::SMALLINT:
			read_func = ReadT<int16_t>;
			break;
		case LogicalTypeId::INTEGER:
			read_func = ReadT<int32_t>;
			break;
		case LogicalTypeId::BIGINT:
			read_func = ReadT<int64_t>;
			break;
		case LogicalTypeId::UTINYINT:
			read_func = ReadT<uint8_t>;
			break;
		case LogicalTypeId::USMALLINT:
			read_func = ReadT<uint16_t>;
			break;
		case LogicalTypeId::UINTEGER:
			read_func = ReadT<uint32_t>;
			break;
		case LogicalTypeId::UBIGINT:
			read_func = ReadT<uint64_t>;
			break;
		default:
			read_func = ReadT<int64_t>;
			break;
		}
	}

	inline int64_t Get(idx_t ridx) const {
		return read_func(data, sel->get_index(ridx));
	}

	inline int64_t GetAt(idx_t idx) const {
		return read_func(data, idx);
	}
};
static inline bool StringEquals(const string_t &a, const string_t &b) noexcept {
	const auto al = a.GetSize();
	if (al != b.GetSize()) {
		return false;
	}
	if (al == 0) {
		return true;
	}
	if (std::memcmp(a.GetPrefix(), b.GetPrefix(), 4) != 0) {
		return false;
	}
	if (al <= 4) {
		return true;
	}
	return std::memcmp(a.GetDataUnsafe(), b.GetDataUnsafe(), al) == 0;
}


static inline size_t HashString(const string_t &s) {
	return duckdb::Hash(s);
}
static inline int CmpStringLex(const string_t &a, const string_t &b) {
	int c = std::memcmp(a.GetPrefix(), b.GetPrefix(), 4);
	if (c != 0) {
		return c;
	}
	const auto al = a.GetSize();
	const auto bl = b.GetSize();
	if (al <= 4 && bl <= 4) {
		return 0;
	}
	const auto ml = std::min(al, bl);
	if (ml > 4) {
		c = std::memcmp(a.GetDataUnsafe() + 4, b.GetDataUnsafe() + 4, ml - 4);
		if (c != 0) {
			return c;
		}
	}
	if (al == bl) {
		return 0;
	}
	return (al < bl) ? -1 : 1;
}


static inline uint64_t PackIds(int32_t ts, int32_t se, int32_t adv) {
	// Use 32 bits for ts, 16 bits for se, 16 bits for adv
	return ((uint64_t)(uint32_t)ts << 32) | ((uint64_t)(uint16_t)se << 16) | (uint64_t)(uint16_t)adv;
}

static inline void UnpackIds(uint64_t packed, int32_t &ts, int32_t &se, int32_t &adv) {
	ts = (int32_t)(uint32_t)(packed >> 32);
	se = (int32_t)(uint16_t)((packed >> 16) & 0xFFFF);
	adv = (int32_t)(uint16_t)(packed & 0xFFFF);
}

// ============================================================
// Key: (Bit-packed IDs, Src, Dst)
// ============================================================

struct HashedKey {
	uint64_t packed_ids;
	string_t src;
	string_t dst;
	size_t hash;

	bool operator==(const HashedKey &o) const noexcept {
		if (hash != o.hash) return false;
		if (packed_ids != o.packed_ids) return false;
		if (!StringEquals(src, o.src)) return false;
		return StringEquals(dst, o.dst);
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept {
		return k.hash;
	}
};

static inline size_t HashCompositeKey(uint64_t packed, const string_t &src, const string_t &dst) {
	size_t h = duckdb::Hash(packed);
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

struct GlobalPartition {
	std::mutex lock;
	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;
};

struct FnGlobalState : public GlobalTableFunctionState {
	GlobalPartition partitions[PARTITION_COUNT];

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
		const uint64_t packed = PackIds(ts, se, adv);
		const size_t h = HashCompositeKey(packed, src_in, dst_in);
		HashedKey probe{packed, src_in, dst_in, h};

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

		map.emplace(HashedKey{packed, src, dst, h}, 1);
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
	IntColumnReader r_ts, r_se, r_adv;
	r_ts.Initialize(t_ts, u_ts);
	r_se.Initialize(t_se, u_se);
	r_adv.Initialize(t_adv, u_adv);

	auto *ref_data = (string_t *)u_ref.data;
	auto *url_data = (string_t *)u_url.data;

	bool cond_const_false = false;
	bool cond_const_true = false;
	if (v_se.GetVectorType() == VectorType::CONSTANT_VECTOR && v_adv.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(v_se) || ConstantVector::IsNull(v_adv)) {
			cond_const_false = true;
		} else {
			int64_t se_val = r_se.GetAt(0);
			int64_t adv_val = r_adv.GetAt(0);
			if (se_val == 0 && adv_val == 0) cond_const_true = true;
			else cond_const_false = true;
		}
	}

	const bool all_valid =
	    u_ts.validity.AllValid() && u_se.validity.AllValid() && u_adv.validity.AllValid() && u_ref.validity.AllValid() && u_url.validity.AllValid();

	if (all_valid) {
		if (cond_const_false) {
			const int32_t se = (int32_t)r_se.GetAt(0);
			const int32_t adv = (int32_t)r_adv.GetAt(0);
			for (idx_t i = 0; i < input.size(); ++i) {
				l.AddOne((int32_t)r_ts.Get(i), se, adv, string_t(), url_data[u_url.sel->get_index(i)]);
			}
		} else if (cond_const_true) {
			const int32_t se = (int32_t)r_se.GetAt(0);
			const int32_t adv = (int32_t)r_adv.GetAt(0);
			for (idx_t i = 0; i < input.size(); ++i) {
				l.AddOne((int32_t)r_ts.Get(i), se, adv, ref_data[u_ref.sel->get_index(i)], url_data[u_url.sel->get_index(i)]);
			}
		} else {
			for (idx_t i = 0; i < input.size(); ++i) {
				const int32_t se = (int32_t)r_se.Get(i);
				const int32_t adv = (int32_t)r_adv.Get(i);
				const string_t src = (se == 0 && adv == 0) ? ref_data[u_ref.sel->get_index(i)] : string_t();
				l.AddOne((int32_t)r_ts.Get(i), se, adv, src, url_data[u_url.sel->get_index(i)]);
			}
		}
	} else {
		for (idx_t i = 0; i < input.size(); ++i) {
			const idx_t ts_i = u_ts.sel->get_index(i);
			const idx_t se_i = u_se.sel->get_index(i);
			const idx_t adv_i = u_adv.sel->get_index(i);
			const idx_t ref_i = u_ref.sel->get_index(i);
			const idx_t url_i = u_url.sel->get_index(i);

			if (!u_ts.validity.RowIsValid(ts_i) || !u_se.validity.RowIsValid(se_i) || 
			    !u_adv.validity.RowIsValid(adv_i) || !u_ref.validity.RowIsValid(ref_i) || 
			    !u_url.validity.RowIsValid(url_i)) continue;

			const int32_t ts = (int32_t)r_ts.GetAt(ts_i);
			const int32_t se = (int32_t)r_se.GetAt(se_i);
			const int32_t adv = (int32_t)r_adv.GetAt(adv_i);
			const string_t referer = ref_data[ref_i];
			const string_t url = url_data[url_i];

			string_t src;
			if (cond_const_false) src = string_t();
			else if (cond_const_true) src = referer;
			else src = (se == 0 && adv == 0) ? referer : string_t();

			l.AddOne(ts, se, adv, src, url);
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
// Partitioned Merge implementation
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	if (local.map.empty()) return;

	struct EntryPtr {
		const HashedKey *key;
		int64_t count;
	};

	const idx_t num_entries = local.map.size();
	std::vector<EntryPtr> all_entries(num_entries);
	
	idx_t counts[PARTITION_COUNT] = {0};
	for (auto it = local.map.begin(); it != local.map.end(); ++it) {
		counts[it->first.hash & (PARTITION_COUNT - 1)]++;
	}

	idx_t offsets[PARTITION_COUNT + 1];
	offsets[0] = 0;
	for (idx_t i = 0; i < PARTITION_COUNT; ++i) {
		offsets[i + 1] = offsets[i] + counts[i];
	}

	idx_t current_offsets[PARTITION_COUNT];
	std::memcpy(current_offsets, offsets, sizeof(idx_t) * PARTITION_COUNT);

	for (auto it = local.map.begin(); it != local.map.end(); ++it) {
		idx_t p_idx = it->first.hash & (PARTITION_COUNT - 1);
		all_entries[current_offsets[p_idx]++] = {&it->first, it->second};
	}

	for (idx_t i = 0; i < PARTITION_COUNT; ++i) {
		if (counts[i] == 0) continue;
		auto &p = g.partitions[i];
		std::lock_guard<std::mutex> guard(p.lock);
		for (idx_t j = offsets[i]; j < offsets[i+1]; ++j) {
			const auto &entry = all_entries[j];
			const HashedKey &k = *entry.key;
			const int64_t cnt = entry.count;

			auto it = p.map.find(k);
			if (it != p.map.end()) {
				it->second += cnt;
			} else {
				string_t g_src = k.src;
				if (g_src.GetSize() != 0) {
					g_src = p.heap.AddString(k.src);
				}
				string_t g_dst = p.heap.AddString(k.dst);
				p.map.emplace(HashedKey{k.packed_ids, g_src, g_dst, k.hash}, cnt);
			}
		}
	}
}

// ============================================================
// Final output logic
// ============================================================

struct TopRow {
	int64_t c;
	uint64_t packed_ids;
	string_t src;
	string_t dst;
	size_t hash;
};

static inline bool BetterRow(const TopRow &a, const TopRow &b) {
	if (a.c != b.c) return a.c > b.c;
	if (a.packed_ids != b.packed_ids) {
		int32_t ats, ase, aadv, bts, bse, badv;
		UnpackIds(a.packed_ids, ats, ase, aadv);
		UnpackIds(b.packed_ids, bts, bse, badv);
		if (ats != bts) return ats < bts;
		if (ase != bse) return ase < bse;
		return aadv < badv;
	}
	int c = CmpStringLex(a.src, b.src);
	if (c != 0) return c < 0;
	c = CmpStringLex(a.dst, b.dst);
	if (c != 0) return c < 0;
	return a.hash < b.hash;
}

struct TopRowWorstCmp {
	bool operator()(const TopRow &a, const TopRow &b) const {
		return BetterRow(a, b);
	}
};

static void EmitOffsetLimitFromPartitions(FnGlobalState &g, DataChunk &out) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowWorstCmp> pq;

	for (idx_t i = 0; i < PARTITION_COUNT; ++i) {
		auto &p = g.partitions[i];
		for (auto &kv : p.map) {
			const HashedKey &k = kv.first;
			const int64_t cnt = kv.second;
			if (cnt <= 0) continue;

			TopRow row;
			row.c = cnt;
			row.packed_ids = k.packed_ids;
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
		int32_t ts, se, adv;
		UnpackIds(top[i].packed_ids, ts, se, adv);
		out_ts[out_idx] = ts;
		out_se[out_idx] = se;
		out_adv[out_idx] = adv;
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
		MergeLocalIntoGlobal(l, g);
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_acq_rel);
	}

	const idx_t merged = g.merged_local_states.load(std::memory_order_acquire);
	const idx_t active = g.active_local_states.load(std::memory_order_acquire);

	if (active > 0 && merged == active) {
		bool expected = false;
		if (g.result_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			EmitOffsetLimitFromPartitions(g, out);
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
