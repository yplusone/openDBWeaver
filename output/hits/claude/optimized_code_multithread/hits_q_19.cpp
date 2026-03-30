/*
query_template: SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) AS cnt
                FROM hits
                GROUP BY UserID, m, SearchPhrase
                ORDER BY COUNT(*) DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT UserID, EventTime, SearchPhrase FROM hits));

query_example: SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) AS cnt
               FROM hits
               GROUP BY UserID, m, SearchPhrase
               ORDER BY COUNT(*) DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT UserID, EventTime, SearchPhrase FROM hits));
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
#include <mutex>
#include <queue>
#include <utility>
#include <vector>
#include <algorithm>

namespace duckdb {

static constexpr size_t NUM_PARTITIONS = 256;

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
	if (al != bl)
		return false;
	if (al == 0)
		return true;
	return std::memcmp(a.GetDataUnsafe(), b.GetDataUnsafe(), al) == 0;
}

static inline int CmpStringLex(const string_t &a, const string_t &b) {
	const auto al = a.GetSize();
	const auto bl = b.GetSize();
	const auto ml = std::min(al, bl);
	if (ml > 0) {
		const int c = std::memcmp(a.GetDataUnsafe(), b.GetDataUnsafe(), ml);
		if (c != 0)
			return c;
	}
	if (al == bl)
		return 0;
	return (al < bl) ? -1 : 1;
}

static inline bool IsTimestampLike(LogicalTypeId id) {
	return id == LogicalTypeId::TIMESTAMP || id == LogicalTypeId::TIMESTAMP_TZ || id == LogicalTypeId::TIMESTAMP_MS ||
	       id == LogicalTypeId::TIMESTAMP_NS || id == LogicalTypeId::TIMESTAMP_SEC;
}

// Read timestamp as INT64 payload (DuckDB stores timestamps as int64). Convert to microseconds.
static inline int64_t ReadTimestampMicrosAt(const UnifiedVectorFormat &uvf, idx_t ridx, LogicalTypeId tid) {
	const idx_t idx = uvf.sel->get_index(ridx);
	const int64_t raw = ((int64_t *)uvf.data)[idx];

	switch (tid) {
	case LogicalTypeId::TIMESTAMP:    // micros
	case LogicalTypeId::TIMESTAMP_TZ: // micros
		return raw;
	case LogicalTypeId::TIMESTAMP_MS: // millis
		return raw * 1000LL;
	case LogicalTypeId::TIMESTAMP_SEC: // seconds
		return raw * 1000000LL;
	case LogicalTypeId::TIMESTAMP_NS: // nanos
		return raw / 1000LL;
	default:
		return raw;
	}
}

static inline int32_t ExtractMinuteFromMicros(int64_t micros) {
	static constexpr int64_t DAY_MICROS = 86400LL * 1000000LL;
	static constexpr int64_t HOUR_MICROS = 3600LL * 1000000LL;
	static constexpr int64_t MINUTE_MICROS = 60LL * 1000000LL;

	int64_t tod = micros % DAY_MICROS;
	if (tod < 0)
		tod += DAY_MICROS;
	int64_t micros_in_hour = tod % HOUR_MICROS;
	return (int32_t)(micros_in_hour / MINUTE_MICROS);
}

// ============================================================
//  Key: (UserID, m, SearchPhrase) with NULL-aware phrase
// ============================================================

struct HashedKey {
	int64_t user_id;
	int32_t minute;
	bool phrase_is_null;
	string_t phrase; // valid if !phrase_is_null
	size_t hash;

	bool operator==(const HashedKey &o) const noexcept {
		if (hash != o.hash)
			return false;
		if (user_id != o.user_id)
			return false;
		if (minute != o.minute)
			return false;
		if (phrase_is_null != o.phrase_is_null)
			return false;
		if (phrase_is_null)
			return true;
		return StringEquals(phrase, o.phrase);
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept { return k.hash; }
};

static inline size_t HashCompositeKey(int64_t user_id, int32_t minute, bool phrase_is_null, const string_t &phrase) {
	size_t h = CombineHash(duckdb::Hash(user_id), duckdb::Hash(minute));
	h = CombineHash(h, duckdb::Hash<uint8_t>(phrase_is_null ? 1 : 0));
	if (!phrase_is_null) {
		h = CombineHash(h, duckdb::Hash(phrase));
	}
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
//  Top10 row materialization (global, stable across finalize calls)
// ============================================================

struct TopRow {
	int64_t cnt;
	int64_t user_id;
	int32_t minute;
	bool phrase_is_null;
	string_t phrase;
	size_t hash;
};

struct TopRowMinCmp {
	// min-heap by cnt
	bool operator()(const TopRow &a, const TopRow &b) const { return a.cnt > b.cnt; }
};

// ============================================================
//  Global / Local state
// ============================================================

struct MapPartition {
	std::mutex lock;
	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;
};

struct FnGlobalState : public GlobalTableFunctionState {
	MapPartition partitions[NUM_PARTITIONS];
	std::mutex merge_lock;

	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};

	// final output cache
	std::vector<TopRow> top10;
	std::atomic<bool> top_built {false};
	std::atomic<idx_t> emit_offset {0};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	StringHeap heap;
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;
	inline void AddOne(int64_t user_id, int32_t minute, const string_t &phrase_in, bool phrase_is_null) {
		const size_t h = HashCompositeKey(user_id, minute, phrase_is_null, phrase_in);
		HashedKey probe {user_id, minute, phrase_is_null, phrase_in, h};

		auto it = map.find(probe);
		if (it != map.end()) {
			it->second += 1;
			return;
		}

		string_t phrase = phrase_is_null ? string_t() : heap.AddString(phrase_in);
		HashedKey heap_key {user_id, minute, phrase_is_null, phrase, h};
		map.emplace(heap_key, 1);
	}

};

// ============================================================
//  Init / Bind
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);
	return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	// MUST match query_template output order:
	// UserID, m, SearchPhrase, cnt
	return_types.push_back(LogicalType::BIGINT);  // UserID
	return_types.push_back(LogicalType::INTEGER); // m
	return_types.push_back(LogicalType::VARCHAR); // SearchPhrase
	return_types.push_back(LogicalType::BIGINT);  // cnt

	names.push_back("UserID");
	names.push_back("m");
	names.push_back("SearchPhrase");
	names.push_back("cnt");

	return make_uniq<FnBindData>();
}

// ============================================================
//  Execute: aggregate into local map
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0)
		return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 3) {
		throw InvalidInputException("dbweaver expects three columns: UserID, EventTime, SearchPhrase");
	}

	auto &v_uid = input.data[0];
	auto &v_time = input.data[1];
	auto &v_phrase = input.data[2];

	const auto t_uid = v_uid.GetType().id();
	const auto t_time = v_time.GetType().id();

	if (!IsIntegral(t_uid)) {
		throw InvalidInputException("dbweaver expects UserID as integral type");
	}
	if (!IsTimestampLike(t_time)) {
		throw InvalidInputException("dbweaver expects EventTime as TIMESTAMP/TIMESTAMP_* type");
	}
	if (v_phrase.GetType().id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("dbweaver expects SearchPhrase as VARCHAR");
	}

	UnifiedVectorFormat u_uid, u_time, u_phrase;
	v_uid.ToUnifiedFormat(input.size(), u_uid);
	v_time.ToUnifiedFormat(input.size(), u_time);
	v_phrase.ToUnifiedFormat(input.size(), u_phrase);

	auto &val_uid = u_uid.validity;
	auto &val_time = u_time.validity;
	auto &val_phrase = u_phrase.validity;

	auto *phrase_data = (string_t *)u_phrase.data;
	const idx_t n = input.size();

	// Type-specialized loops for BIGINT and TIMESTAMP/TZ (micros)
	if (t_uid == LogicalTypeId::BIGINT && (t_time == LogicalTypeId::TIMESTAMP || t_time == LogicalTypeId::TIMESTAMP_TZ)) {
		auto *uid_data = (int64_t *)u_uid.data;
		auto *time_data = (int64_t *)u_time.data;
		for (idx_t ridx = 0; ridx < n; ridx++) {
			const idx_t uid_i = u_uid.sel->get_index(ridx);
			const idx_t time_i = u_time.sel->get_index(ridx);

			if (!val_uid.RowIsValid(uid_i) || !val_time.RowIsValid(time_i))
				continue;

			const int64_t uid = uid_data[uid_i];
			const int32_t m = ExtractMinuteFromMicros(time_data[time_i]);

			const idx_t ph_i = u_phrase.sel->get_index(ridx);
			const bool phrase_is_null = !val_phrase.RowIsValid(ph_i);
			const string_t phrase = phrase_is_null ? string_t() : phrase_data[ph_i];

			l.AddOne(uid, m, phrase, phrase_is_null);
		}
	} else {
		// Generic path fallback
		for (idx_t ridx = 0; ridx < n; ridx++) {
			const idx_t uid_i = u_uid.sel->get_index(ridx);
			const idx_t time_i = u_time.sel->get_index(ridx);

			if (!val_uid.RowIsValid(uid_i) || !val_time.RowIsValid(time_i))
				continue;

			const int64_t uid = ReadInt64At(u_uid, ridx, t_uid);
			const int64_t micros = ReadTimestampMicrosAt(u_time, ridx, t_time);
			const int32_t m = ExtractMinuteFromMicros(micros);

			const idx_t ph_i = u_phrase.sel->get_index(ridx);
			const bool phrase_is_null = !val_phrase.RowIsValid(ph_i);
			const string_t phrase = phrase_is_null ? string_t() : phrase_data[ph_i];

			l.AddOne(uid, m, phrase, phrase_is_null);
		}
	}


	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Merge helper (Partitioned)
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (auto it = local.map.begin(); it != local.map.end(); ++it) {
		const HashedKey &k = it->first;
		const int64_t cnt = it->second;
		if (cnt <= 0)
			continue;

		// Determine partition
		size_t p_idx = k.hash % NUM_PARTITIONS;
		auto &part = g.partitions[p_idx];

		std::lock_guard<std::mutex> guard(part.lock);
		auto g_it = part.map.find(k);
		if (g_it != part.map.end()) {
			g_it->second += cnt;
			continue;
		}

		// materialize phrase into partition heap
		string_t g_phrase = k.phrase;
		if (!k.phrase_is_null) {
			g_phrase = part.heap.AddString(k.phrase);
		} else {
			g_phrase = string_t();
		}

		HashedKey g_key = k;
		g_key.phrase = g_phrase;
		part.map.emplace(g_key, cnt);
	}
}

// ============================================================
//  Build Top10 once (after all locals merged)
// ============================================================

static inline void BuildTop10(FnGlobalState &g) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (size_t i = 0; i < NUM_PARTITIONS; ++i) {
		auto &part = g.partitions[i];
		for (auto it = part.map.begin(); it != part.map.end(); ++it) {
			const HashedKey &k = it->first;
			const int64_t cnt = it->second;
			if (cnt <= 0)
				continue;

			TopRow row;
			row.cnt = cnt;
			row.user_id = k.user_id;
			row.minute = k.minute;
			row.phrase_is_null = k.phrase_is_null;
			row.phrase = k.phrase;
			row.hash = k.hash;

			if (pq.size() < 10) {
				pq.push(row);
			} else if (row.cnt > pq.top().cnt) {
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

	// ORDER BY cnt DESC, then deterministic tie-breaks
	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.cnt != b.cnt)
			return a.cnt > b.cnt;
		if (a.user_id != b.user_id)
			return a.user_id < b.user_id;
		if (a.minute != b.minute)
			return a.minute < b.minute;
		if (a.phrase_is_null != b.phrase_is_null)
			return a.phrase_is_null < b.phrase_is_null; // non-null first
		if (a.phrase_is_null)
			return false;
		const int c = CmpStringLex(a.phrase, b.phrase);
		if (c != 0)
			return c < 0;
		return a.hash < b.hash;
	});

	g.top10 = std::move(top);
	g.emit_offset.store(0, std::memory_order_relaxed);
	g.top_built.store(true, std::memory_order_release);
}

// ============================================================
//  Finalize: merge, wait-all, emit top10 only
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	// merge local exactly once into partitioned global
	if (!l.merged) {
		MergeLocalIntoGlobal(l, g);
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_acq_rel);
	}

	const auto merged = g.merged_local_states.load(std::memory_order_acquire);
	const auto active = g.active_local_states.load(std::memory_order_acquire);

	// not the last finisher: emit nothing
	if (active == 0 || merged < active) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	// last finisher builds top10 once
	if (!g.top_built.load(std::memory_order_acquire)) {
		std::lock_guard<std::mutex> guard(g.merge_lock);
		if (!g.top_built.load(std::memory_order_acquire)) {
			BuildTop10(g);
		}
	}

	// emit cached top10 (single chunk is enough, but keep it robust)
	const idx_t offset = g.emit_offset.load(std::memory_order_relaxed);
	if (offset >= g.top10.size()) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	const idx_t remaining = (idx_t)g.top10.size() - offset;
	const idx_t emit_n = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);

	auto *out_uid = FlatVector::GetData<int64_t>(out.data[0]);
	auto *out_m = FlatVector::GetData<int32_t>(out.data[1]);
	auto *out_phrase = FlatVector::GetData<string_t>(out.data[2]);
	auto *out_cnt = FlatVector::GetData<int64_t>(out.data[3]);

	for (idx_t i = 0; i < emit_n; i++) {
		const TopRow &r = g.top10[offset + i];

		out_uid[i] = r.user_id;
		out_m[i] = r.minute;

		if (r.phrase_is_null) {
			FlatVector::SetNull(out.data[2], i, true);
		} else {
			out_phrase[i] = r.phrase;
		}

		out_cnt[i] = r.cnt;
	}

	out.SetCardinality(emit_n);
	g.emit_offset.fetch_add(emit_n, std::memory_order_relaxed);

	// if more output remains, tell DuckDB
	if (g.emit_offset.load(std::memory_order_relaxed) < g.top10.size()) {
		return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
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