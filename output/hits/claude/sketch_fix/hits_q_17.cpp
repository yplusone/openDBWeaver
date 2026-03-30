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
//  Global state
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;

	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};

	// 0:not started, 1:adopting, 2:adopt done
	std::atomic<uint8_t> adopt_stage{0};

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
		string_t phrase = phrase_in;
		if (!phrase_is_null) {
			phrase = heap.AddString(phrase_in); // stable across chunks
		} else {
			phrase = string_t(); // unused
		}

		const size_t h = HashCompositeKey(user_id, phrase, phrase_is_null);
		HashedKey probe{user_id, phrase_is_null, phrase, h};

		auto it = map.find(probe);
		if (it != map.end()) {
			it->second += 1;
			return;
		}
		map.emplace(probe, 1);
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

	constexpr idx_t BATCH = 8;

	for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
		const idx_t n = std::min(BATCH, input.size() - rr);
		for (idx_t k = 0; k < n; ++k) {
			const idx_t ridx = rr + k;

			const idx_t uid_i = u_uid.sel->get_index(ridx);
			if (!val_uid.RowIsValid(uid_i)) continue; // NULL user_id -> skip (SQL would group NULL; adjust if needed)

			const int64_t uid = ReadInt64At(u_uid, ridx, t_uid);

			const idx_t ph_i = u_phrase.sel->get_index(ridx);
			const bool phrase_is_null = !val_phrase.RowIsValid(ph_i);
			const string_t phrase = phrase_is_null ? string_t() : phrase_data[ph_i];

			l.AddOne(uid, phrase, phrase_is_null);
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Merge helper
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (auto &kv : local.map) {
		const HashedKey &k = kv.first;
		const int64_t cnt = kv.second;
		if (cnt <= 0) continue;

		auto it = g.map.find(k);
		if (it != g.map.end()) {
			it->second += cnt;
			continue;
		}

		// Copy phrase into global heap if non-NULL.
		string_t g_phrase = k.phrase;
		if (!k.phrase_is_null) {
			g_phrase = g.heap.AddString(k.phrase);
		} else {
			g_phrase = string_t();
		}

		const size_t h = HashCompositeKey(k.user_id, g_phrase, k.phrase_is_null);
		g.map.emplace(HashedKey{k.user_id, k.phrase_is_null, g_phrase, h}, cnt);
	}
}

// ============================================================
//  Finalize: Top10 by cnt DESC
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
static void EmitTop10(DataChunk &out, absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> &map_ref) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (auto &kv : map_ref) {
		const HashedKey &k = kv.first;
		const int64_t cnt = kv.second;
		if (cnt <= 0) {
			continue;
		}

		TopRow row{cnt, k.user_id, k.phrase_is_null, k.phrase, k.hash};

		if (pq.size() < 10) {
			pq.push(row);
		} else if (row.cnt > pq.top().cnt) {
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
		if (a.cnt != b.cnt) {
			return a.cnt > b.cnt;
		}
		if (a.user_id != b.user_id) {
			return a.user_id < b.user_id;
		}
		if (a.phrase_is_null != b.phrase_is_null) {
			// non-null first
			return a.phrase_is_null < b.phrase_is_null;
		}
		if (a.phrase_is_null) {
			return false;
		}
		const int c = CmpStringLex(a.phrase, b.phrase);
		if (c != 0) {
			return c < 0;
		}
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

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active == 1) {
		EmitTop10(out, l.map);
		return OperatorFinalizeResultType::FINISHED;
	}

	// merge
	if (!l.merged) {
		uint8_t expected = 0;
		if (g.adopt_stage.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> guard(g.lock);

			// Always merge via helper to avoid assigning StringHeap
			MergeLocalIntoGlobal(l, g);

			g.adopt_stage.store(2, std::memory_order_release);
			l.merged = true;
			g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
		} else {
			while (g.adopt_stage.load(std::memory_order_acquire) == 1) {
				std::this_thread::yield();
			}
			std::lock_guard<std::mutex> guard(g.lock);
			MergeLocalIntoGlobal(l, g);
			l.merged = true;
			g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
		}
	}

	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);

	// Only one finalize should output.
	if (!(active > 0 && merged == active)) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	EmitTop10(out, g.map);
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