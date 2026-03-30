/*
query_template: SELECT UserID, SearchPhrase, COUNT(*) AS cnt
                FROM hits
                GROUP BY UserID, SearchPhrase
                LIMIT 10;

split_template: select * from dbweaver((SELECT UserID, SearchPhrase FROM hits));

query_example: SELECT UserID, SearchPhrase, COUNT(*) AS cnt
               FROM hits
               GROUP BY UserID, SearchPhrase
               LIMIT 10;

split_query: select * from dbweaver((SELECT UserID, SearchPhrase FROM hits));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/table_function.hpp"

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace duckdb {

// ============================================================
// Helpers
// ============================================================

static constexpr idx_t NUM_SHARDS = 64; // must be power of two

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

static inline idx_t ShardIndex(hash_t h) {
	return ((idx_t)h) & (NUM_SHARDS - 1);
}

// ============================================================
// Key: (UserID, SearchPhrase) with NULL-aware fields
// ============================================================

struct HashedKey {
	bool user_is_null;
	int64_t user_id; // valid if !user_is_null

	bool phrase_is_null;
	string_t phrase; // valid if !phrase_is_null

	hash_t hash;

	bool operator==(const HashedKey &o) const noexcept {
		if (hash != o.hash) {
			return false;
		}
		if (user_is_null != o.user_is_null) {
			return false;
		}
		if (!user_is_null && user_id != o.user_id) {
			return false;
		}
		if (phrase_is_null != o.phrase_is_null) {
			return false;
		}
		if (phrase_is_null) {
			return true;
		}
		return StringEquals(phrase, o.phrase);
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept {
		return (size_t)k.hash;
	}
};

// ============================================================
// Bind data
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
// Sharded states
// ============================================================

struct GlobalShard {
	std::mutex lock;
	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;
};

struct LocalShard {
	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;
};

// ============================================================
// Global state
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	std::unique_ptr<GlobalShard[]> shards;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};
	std::atomic<bool> output_emitted{false};

	FnGlobalState() : shards(make_uniq_array<GlobalShard>(NUM_SHARDS)) {
	}

	idx_t MaxThreads() const override {
		return std::numeric_limits<idx_t>::max();
	}
};

// ============================================================
// Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	std::unique_ptr<LocalShard[]> shards;

	FnLocalState() : shards(make_uniq_array<LocalShard>(NUM_SHARDS)) {
	}

	inline void AddOne(bool user_is_null, int64_t user_id, bool phrase_is_null, const string_t &phrase_in, hash_t h) {
		auto &shard = shards[ShardIndex(h)];

		HashedKey probe{user_is_null, user_id, phrase_is_null, phrase_in, h};
		auto it = shard.map.find(probe);
		if (it != shard.map.end()) {
			it->second += 1;
			return;
		}

		string_t phrase = phrase_in;
		if (!phrase_is_null) {
			phrase = shard.heap.AddString(phrase_in);
		} else {
			phrase = string_t();
		}

		shard.map.emplace(HashedKey{user_is_null, user_id, phrase_is_null, phrase, h}, 1);
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
                                       vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::BIGINT);  // UserID
	return_types.push_back(LogicalType::VARCHAR); // SearchPhrase
	return_types.push_back(LogicalType::BIGINT);  // cnt

	names.push_back("UserID");
	names.push_back("SearchPhrase");
	names.push_back("cnt");

	return make_uniq<FnBindData>();
}

// ============================================================
// Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &context, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	const idx_t size = input.size();
	if (size == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	auto &l = in.local_state->Cast<FnLocalState>();

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

	Vector hashes(LogicalType::HASH, size);
	if (t_uid == LogicalTypeId::BIGINT) {
		VectorOperations::Hash(v_uid, hashes, size);
	} else {
		Vector v_uid_bigint(LogicalType::BIGINT, size);
		VectorOperations::Cast(context.client, v_uid, v_uid_bigint, size);
		VectorOperations::Hash(v_uid_bigint, hashes, size);
	}
	VectorOperations::CombineHash(hashes, v_phrase, size);

	UnifiedVectorFormat u_uid, u_phrase, u_hashes;
	v_uid.ToUnifiedFormat(size, u_uid);
	v_phrase.ToUnifiedFormat(size, u_phrase);
	hashes.ToUnifiedFormat(size, u_hashes);

	auto *phrase_data = (string_t *)u_phrase.data;
	auto *hash_ptr = (hash_t *)u_hashes.data;

	for (idx_t i = 0; i < size; ++i) {
		const idx_t uid_i = u_uid.sel->get_index(i);
		const bool user_is_null = !u_uid.validity.RowIsValid(uid_i);
		const int64_t uid = user_is_null ? 0 : ReadInt64At(u_uid, i, t_uid);

		const idx_t ph_i = u_phrase.sel->get_index(i);
		const bool phrase_is_null = !u_phrase.validity.RowIsValid(ph_i);
		const string_t phrase = phrase_is_null ? string_t() : phrase_data[ph_i];

		const hash_t h = hash_ptr[u_hashes.sel->get_index(i)];

		l.AddOne(user_is_null, uid, phrase_is_null, phrase, h);
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
// Merge helpers
// ============================================================

static inline void MergeOneShard(LocalShard &local_shard, GlobalShard &global_shard) {
	if (local_shard.map.empty()) {
		return;
	}

	// Reduce rehashes during merge.
	global_shard.map.reserve(global_shard.map.size() + local_shard.map.size());

	for (auto &kv : local_shard.map) {
		const HashedKey &k = kv.first;
		const int64_t cnt = kv.second;
		if (cnt <= 0) {
			continue;
		}

		auto it = global_shard.map.find(k);
		if (it != global_shard.map.end()) {
			it->second += cnt;
			continue;
		}

		string_t g_phrase = k.phrase;
		if (!k.phrase_is_null) {
			g_phrase = global_shard.heap.AddString(k.phrase);
		} else {
			g_phrase = string_t();
		}

		global_shard.map.emplace(HashedKey{k.user_is_null, k.user_id, k.phrase_is_null, g_phrase, k.hash}, cnt);
	}
}

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (idx_t s = 0; s < NUM_SHARDS; ++s) {
		auto &local_shard = local.shards[s];
		if (local_shard.map.empty()) {
			continue;
		}

		auto &global_shard = g.shards[s];
		std::lock_guard<std::mutex> guard(global_shard.lock);
		MergeOneShard(local_shard, global_shard);
	}
}

// ============================================================
// Emit helpers
// ============================================================

static idx_t EmitAny10FromLocal(FnLocalState &l, DataChunk &out) {
	auto *out_uid = FlatVector::GetData<int64_t>(out.data[0]);
	auto *out_cnt = FlatVector::GetData<int64_t>(out.data[2]);

	idx_t out_idx = 0;
	for (idx_t s = 0; s < NUM_SHARDS && out_idx < 10; ++s) {
		for (auto it = l.shards[s].map.begin(); it != l.shards[s].map.end() && out_idx < 10; ++it) {
			const HashedKey &k = it->first;
			const int64_t cnt = it->second;

			out_cnt[out_idx] = cnt;

			if (k.user_is_null) {
				FlatVector::SetNull(out.data[0], out_idx, true);
			} else {
				out_uid[out_idx] = k.user_id;
			}

			if (k.phrase_is_null) {
				FlatVector::SetNull(out.data[1], out_idx, true);
			} else {
				FlatVector::SetNull(out.data[1], out_idx, false);
				FlatVector::GetData<string_t>(out.data[1])[out_idx] =
				    StringVector::AddString(out.data[1], k.phrase);
			}

			++out_idx;
		}
	}
	out.SetCardinality(out_idx);
	return out_idx;
}

static idx_t EmitAny10FromGlobal(FnGlobalState &g, DataChunk &out) {
	auto *out_uid = FlatVector::GetData<int64_t>(out.data[0]);
	auto *out_cnt = FlatVector::GetData<int64_t>(out.data[2]);

	idx_t out_idx = 0;
	for (idx_t s = 0; s < NUM_SHARDS && out_idx < 10; ++s) {
		auto &global_shard = g.shards[s];
		for (auto it = global_shard.map.begin(); it != global_shard.map.end() && out_idx < 10; ++it) {
			const HashedKey &k = it->first;
			const int64_t cnt = it->second;

			out_cnt[out_idx] = cnt;

			if (k.user_is_null) {
				FlatVector::SetNull(out.data[0], out_idx, true);
			} else {
				out_uid[out_idx] = k.user_id;
			}

			if (k.phrase_is_null) {
				FlatVector::SetNull(out.data[1], out_idx, true);
			} else {
				FlatVector::SetNull(out.data[1], out_idx, false);
				FlatVector::GetData<string_t>(out.data[1])[out_idx] =
				    StringVector::AddString(out.data[1], k.phrase);
			}

			++out_idx;
		}
	}
	out.SetCardinality(out_idx);
	return out_idx;
}

// ============================================================
// Finalize: LIMIT 10 (no ORDER BY) => emit any 10 groups
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_acquire);

	// single-thread fast path: no global merge needed
	if (active == 1) {
		EmitAny10FromLocal(l, out);
		return OperatorFinalizeResultType::FINISHED;
	}

	if (!l.merged) {
		MergeLocalIntoGlobal(l, g);
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_acq_rel);
	}

	const auto merged = g.merged_local_states.load(std::memory_order_acquire);
	if (merged != active) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	bool expected = false;
	if (!g.output_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	EmitAny10FromGlobal(g, out);
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