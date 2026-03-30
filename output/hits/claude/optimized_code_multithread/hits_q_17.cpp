/*
query_template: SELECT UserID, SearchPhrase, COUNT(*) AS cnt
                FROM hits
                GROUP BY UserID, SearchPhrase
                ORDER BY COUNT(*) DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT UserID, SearchPhrase FROM hits));

query_example: SELECT UserID, SearchPhrase, COUNT(*) AS cnt
               FROM hits
               GROUP BY UserID, SearchPhrase
               ORDER BY COUNT(*) DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT UserID, SearchPhrase FROM hits));
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
#include <thread>
#include <utility>
#include <vector>
#include <algorithm>

namespace duckdb {

// ============================================================
//  Helpers
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
	if (al != bl) return false;
	if (al == 0) return true;
	return std::memcmp(a.GetDataUnsafe(), b.GetDataUnsafe(), al) == 0;
}

static inline int CmpStringLex(const string_t &a, const string_t &b) {
	const auto al = a.GetSize();
	const auto bl = b.GetSize();
	const auto ml = std::min(al, bl);
	if (ml > 0) {
		const int c = std::memcmp(a.GetDataUnsafe(), b.GetDataUnsafe(), ml);
		if (c != 0) return c;
	}
	if (al == bl) return 0;
	return (al < bl) ? -1 : 1;
}

// ============================================================
//  Key: (UserID, SearchPhrase) with NULL-aware phrase
// ============================================================

struct HashedKey {
	int64_t user_id;
	bool phrase_is_null;
	string_t phrase; // only valid when phrase_is_null == false
	size_t hash;

	bool operator==(const HashedKey &o) const noexcept {
		if (hash != o.hash) return false;
		if (user_id != o.user_id) return false;
		if (phrase_is_null != o.phrase_is_null) return false;
		if (phrase_is_null) return true;
		return StringEquals(phrase, o.phrase);
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept { return k.hash; }
};

static inline size_t HashPhraseNullable(const string_t &s, bool is_null) {
	// Distinguish NULL from empty string by mixing a constant.
	size_t h = is_null ? duckdb::Hash<uint8_t>(1) : duckdb::Hash<uint8_t>(0);
	if (!is_null) {
		h = CombineHash(h, duckdb::Hash(s)); // content-based
	}
	return h;
}

static inline size_t HashCompositeKey(int64_t user_id, const string_t &phrase, bool phrase_is_null) {
	size_t h = CombineHash(duckdb::Hash(user_id), HashPhraseNullable(phrase, phrase_is_null));
	return h;
}

// ============================================================
//  Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

// ============================================================
//  Global state (Partitioned)
// ============================================================

struct GlobalPartition {
	std::mutex lock;
	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;
};

struct FnGlobalState : public GlobalTableFunctionState {
	static constexpr idx_t PARTITION_COUNT = 128;
	std::vector<std::unique_ptr<GlobalPartition>> partitions;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};

	FnGlobalState() {
		for (idx_t i = 0; i < PARTITION_COUNT; ++i) {
			partitions.push_back(make_uniq<GlobalPartition>());
		}
	}

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

// ============================================================
//  Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;
	inline void AddOne(int64_t user_id, const string_t &phrase_in, bool phrase_is_null) {
		const size_t h = HashCompositeKey(user_id, phrase_in, phrase_is_null);
		HashedKey probe {user_id, phrase_is_null, phrase_in, h};

		auto it = map.find(probe);
		if (it != map.end()) {
			it->second += 1;
			return;
		}

		string_t phrase = phrase_is_null ? string_t() : heap.AddString(phrase_in);
		map.emplace(HashedKey {user_id, phrase_is_null, phrase, h}, 1);
	}
};

// ============================================================
//  Finalize Helpers
// ============================================================

struct TopRow {
	int64_t cnt;
	int64_t user_id;
	bool phrase_is_null;
	string_t phrase;
	size_t hash;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.cnt > b.cnt; } // min-heap by cnt
};

static void SortAndEmit(DataChunk &out, std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq) {
	std::vector<TopRow> top;
	top.reserve(pq.size());
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}

	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.cnt != b.cnt) return a.cnt > b.cnt;
		if (a.user_id != b.user_id) return a.user_id < b.user_id;
		if (a.phrase_is_null != b.phrase_is_null) return a.phrase_is_null < b.phrase_is_null;
		if (a.phrase_is_null) return false;
		const int c = CmpStringLex(a.phrase, b.phrase);
		if (c != 0) return c < 0;
		return a.hash < b.hash;
	});

	auto *out_uid = FlatVector::GetData<int64_t>(out.data[0]);
	auto *out_phrase = FlatVector::GetData<string_t>(out.data[1]);
	auto *out_cnt = FlatVector::GetData<int64_t>(out.data[2]);

	idx_t out_idx = 0;
	for (auto &r : top) {
		out_uid[out_idx] = r.user_id;
		out_cnt[out_idx] = r.cnt;
		if (r.phrase_is_null) {
			FlatVector::SetNull(out.data[1], out_idx, true);
		} else {
			out_phrase[out_idx] = r.phrase;
		}
		++out_idx;
	}
	out.SetCardinality(out_idx);
}

static void EmitTop10(DataChunk &out, absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> &map_ref) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	for (auto &kv : map_ref) {
		const HashedKey &k = kv.first;
		const int64_t cnt = kv.second;
		if (cnt <= 0) continue;
		TopRow row{cnt, k.user_id, k.phrase_is_null, k.phrase, k.hash};
		if (pq.size() < 10) {
			pq.push(row);
		} else if (row.cnt > pq.top().cnt) {
			pq.pop();
			pq.push(row);
		}
	}
	SortAndEmit(out, pq);
}

static void EmitTop10Global(DataChunk &out, FnGlobalState &g) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	for (idx_t i = 0; i < FnGlobalState::PARTITION_COUNT; ++i) {
		auto &map_ref = g.partitions[i]->map;
		for (auto &kv : map_ref) {
			const HashedKey &k = kv.first;
			const int64_t cnt = kv.second;
			if (cnt <= 0) continue;
			TopRow row{cnt, k.user_id, k.phrase_is_null, k.phrase, k.hash};
			if (pq.size() < 10) {
				pq.push(row);
			} else if (row.cnt > pq.top().cnt) {
				pq.pop();
				pq.push(row);
			}
		}
	}
	SortAndEmit(out, pq);
}

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
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	return_types.push_back(LogicalType::BIGINT);  // UserID
	return_types.push_back(LogicalType::VARCHAR); // SearchPhrase
	return_types.push_back(LogicalType::BIGINT);  // cnt

	names.push_back("UserID");
	names.push_back("SearchPhrase");
	names.push_back("cnt");

	return make_uniq<FnBindData>();
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();

	// Expect 2 columns: UserID, SearchPhrase
	if (input.ColumnCount() < 2) {
		throw InvalidInputException("dbweaver expects two columns: UserID, SearchPhrase");
	}

	auto &v_uid = input.data[0];
	auto &v_phrase = input.data[1];

	const auto t_uid = v_uid.GetType().id();
	if (!IsIntegral(t_uid)) {
		throw InvalidInputException("dbweaver expects UserID as integral type");
	}
	if (v_phrase.GetType().id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("dbweaver expects SearchPhrase as VARCHAR");
	}

	UnifiedVectorFormat u_uid, u_phrase;
	v_uid.ToUnifiedFormat(input.size(), u_uid);
	v_phrase.ToUnifiedFormat(input.size(), u_phrase);

	auto &val_uid = u_uid.validity;
	auto &val_phrase = u_phrase.validity;

	auto *phrase_data = (string_t *)u_phrase.data;

	for (idx_t ridx = 0; ridx < input.size(); ++ridx) {
		const idx_t uid_i = u_uid.sel->get_index(ridx);
		if (!val_uid.RowIsValid(uid_i)) continue; // skip NULL user_id

		const int64_t uid = ReadInt64At(u_uid, ridx, t_uid);

		const idx_t ph_i = u_phrase.sel->get_index(ridx);
		const bool phrase_is_null = !val_phrase.RowIsValid(ph_i);
		const string_t phrase = phrase_is_null ? string_t() : phrase_data[ph_i];

		l.AddOne(uid, phrase, phrase_is_null);
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize: Partitioned Merge + Top10
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);
	if (active == 0) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	// Single-threaded optimization
	if (active == 1) {
		EmitTop10(out, l.map);
		return OperatorFinalizeResultType::FINISHED;
	}
	// Parallel Partitioned Merge
	if (!l.merged) {
		// Group local entries into partition buckets to reduce locking frequency
		std::vector<std::pair<HashedKey, int64_t>> buckets[FnGlobalState::PARTITION_COUNT];
		for (const auto &kv : l.map) {
			idx_t p_idx = kv.first.hash % FnGlobalState::PARTITION_COUNT;
			buckets[p_idx].push_back(kv);
		}

		for (idx_t p_idx = 0; p_idx < FnGlobalState::PARTITION_COUNT; ++p_idx) {
			auto &bucket = buckets[p_idx];
			if (bucket.empty()) {
				continue;
			}

			GlobalPartition &part = *g.partitions[p_idx];
			std::lock_guard<std::mutex> guard(part.lock);
			for (const auto &kv : bucket) {
				const HashedKey &k = kv.first;
				auto it = part.map.find(k);
				if (it != part.map.end()) {
					it->second += kv.second;
				} else {
					string_t g_phrase = k.phrase;
					if (!k.phrase_is_null) {
						g_phrase = part.heap.AddString(k.phrase);
					}
					part.map.emplace(HashedKey{k.user_id, k.phrase_is_null, g_phrase, k.hash}, kv.second);
				}
			}
		}
		l.merged = true;


		const idx_t finished = g.merged_local_states.fetch_add(1, std::memory_order_acq_rel) + 1;
		if (finished == active) {
			EmitTop10Global(out, g);
		} else {
			out.SetCardinality(0);
		}
	} else {
		out.SetCardinality(0);
	}

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

void DbweaverExtension::Load(ExtensionLoader &loader) { LoadInternal(loader); }
std::string DbweaverExtension::Name() { return "dbweaver"; }
std::string DbweaverExtension::Version() const { return DuckDB::LibraryVersion(); }

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) {
	duckdb::LoadInternal(loader);
}

}
