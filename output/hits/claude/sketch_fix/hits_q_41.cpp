/*
query_template: SELECT URLHash, EventDate, COUNT(*) AS PageViews
                FROM hits
                WHERE CounterID = 62
                  AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                  AND IsRefresh = 0
                  AND TraficSourceID IN (-1, 6)
                  AND RefererHash = 3594120000172545465
                GROUP BY URLHash, EventDate
                ORDER BY PageViews DESC
                LIMIT 10 OFFSET 100;

split_template: select * from dbweaver((SELECT EventDate, TraficSourceID, URLHash
                                       FROM hits
                                       WHERE (CounterID=62)
                                         AND (IsRefresh=0)
                                         AND (optional: TraficSourceID IN (-1, 6))
                                         AND (RefererHash=3594120000172545465)));

query_example: SELECT URLHash, EventDate, COUNT(*) AS PageViews
               FROM hits
               WHERE CounterID = 62
                 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                 AND IsRefresh = 0
                 AND TraficSourceID IN (-1, 6)
                 AND RefererHash = 3594120000172545465
               GROUP BY URLHash, EventDate
               ORDER BY PageViews DESC
               LIMIT 10 OFFSET 100;

split_query: select * from dbweaver((SELECT EventDate, TraficSourceID, URLHash
                                    FROM hits
                                    WHERE (CounterID=62)
                                      AND (IsRefresh=0)
                                      AND (optional: TraficSourceID IN (-1, 6))
                                      AND (RefererHash=3594120000172545465)));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <cstdint>
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
//  Constants for this query
// ============================================================

static constexpr idx_t TOPK_LIMIT = 10;
static constexpr idx_t TOPK_OFFSET = 100;
static constexpr idx_t TOPK_NEED = TOPK_LIMIT + TOPK_OFFSET; // 110

// ============================================================
//  Key: (URLHash, EventDate) + precomputed hash
// ============================================================

struct HashedKey {
	int64_t url_hash;
	int32_t event_date_days; 
	size_t hash;

	bool operator==(const HashedKey &o) const noexcept {
		return hash == o.hash && url_hash == o.url_hash && event_date_days == o.event_date_days;
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept { return k.hash; }
};

static inline size_t HashCompositeKey(int64_t url_hash, int32_t event_date_days) {
	return CombineHash(duckdb::Hash(url_hash), duckdb::Hash(event_date_days));
}

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

static inline int32_t ReadDate32At(const UnifiedVectorFormat &uvf, idx_t ridx, LogicalTypeId tid) {
	const idx_t idx = uvf.sel->get_index(ridx);
	if (tid == LogicalTypeId::DATE) {
		return ((date_t *)uvf.data)[idx].days;
	}
	return (int32_t)ReadInt64At(uvf, ridx, tid);
}

// ============================================================
//  Global / Local States
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};
struct FnGlobalState : public GlobalTableFunctionState {
	absl::flat_hash_map<HashedKey, int64_t, HashedKeyHash> map;
	std::mutex lock;
	std::atomic<bool> produced_result{false};
	idx_t MaxThreads() const override { return 1; }
};



struct FnLocalState : public LocalTableFunctionState {};

// ============================================================
//  Init / Bind
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &, GlobalTableFunctionState *) {
	return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::BIGINT); // URLHash
	return_types.push_back(LogicalType::DATE);   // EventDate
	return_types.push_back(LogicalType::BIGINT); // PageViews
	names.push_back("URLHash");
	names.push_back("EventDate");
	names.push_back("PageViews");
	return make_uniq<FnBindData>();
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &g = in.global_state->Cast<FnGlobalState>();
	
	// Input indices: EventDate(0), TraficSourceID(1), URLHash(2)
	auto &v_date = input.data[0];
	auto &v_ts = input.data[1];
	auto &v_urlhash = input.data[2];

	const auto t_date = v_date.GetType().id();
	const auto t_ts = v_ts.GetType().id();
	const auto t_urlhash = v_urlhash.GetType().id();

	UnifiedVectorFormat u_date, u_ts, u_urlhash;
	v_date.ToUnifiedFormat(input.size(), u_date);
	v_ts.ToUnifiedFormat(input.size(), u_ts);
	v_urlhash.ToUnifiedFormat(input.size(), u_urlhash);

	std::lock_guard<std::mutex> guard(g.lock);
	for (idx_t i = 0; i < input.size(); i++) {

		const idx_t date_i = u_date.sel->get_index(i);
		const idx_t ts_i = u_ts.sel->get_index(i);
		const idx_t url_i = u_urlhash.sel->get_index(i);

		if (!u_date.validity.RowIsValid(date_i) || !u_ts.validity.RowIsValid(ts_i) || !u_urlhash.validity.RowIsValid(url_i)) {
			continue;
		}
		const int32_t ts = (int32_t)ReadInt64At(u_ts, i, t_ts);
		if (!(ts == -1 || ts == 6)) continue;

		const int32_t date_days = ReadDate32At(u_date, i, t_date);
		if (date_days < 15887 || date_days > 15917) continue;

		const int64_t url_hash = ReadInt64At(u_urlhash, i, t_urlhash);

		const size_t h = HashCompositeKey(url_hash, date_days);

		HashedKey key{url_hash, date_days, h};
		g.map[key] += 1;
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize
// ============================================================

struct TopRow {
	int64_t c;
	int64_t url_hash;
	int32_t event_date_days;
};
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	if (g.produced_result.exchange(true)) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	std::lock_guard<std::mutex> guard(g.lock);
	if (g.map.empty()) {


		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	std::vector<TopRow> top;
	top.reserve(g.map.size());
	for (auto const &kv : g.map) {
		top.push_back({kv.second, kv.first.url_hash, kv.first.event_date_days});
	}

	// ORDER BY PageViews DESC, EventDate ASC, URLHash ASC
	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.c != b.c) return a.c > b.c;
		if (a.event_date_days != b.event_date_days) return a.event_date_days < b.event_date_days;
		return a.url_hash < b.url_hash;
	});

	if (top.size() <= TOPK_OFFSET) {
		out.SetCardinality(0);
	} else {
		const idx_t start = TOPK_OFFSET;
		const idx_t end = std::min<idx_t>(top.size(), TOPK_OFFSET + TOPK_LIMIT);
		idx_t out_idx = 0;
		int64_t *out_urlhash = FlatVector::GetData<int64_t>(out.data[0]);
		date_t *out_date = FlatVector::GetData<date_t>(out.data[1]);
		int64_t *out_pv = FlatVector::GetData<int64_t>(out.data[2]);

		for (idx_t i = start; i < end; ++i) {
			out_urlhash[out_idx] = top[i].url_hash;
			out_date[out_idx] = date_t(top[i].event_date_days);
			out_pv[out_idx] = top[i].c;
			++out_idx;
		}
		out.SetCardinality(out_idx);
	}

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