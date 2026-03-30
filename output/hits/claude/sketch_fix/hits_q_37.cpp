/*
query_template: SELECT URL, COUNT(*) AS PageViews
                FROM hits
                WHERE CounterID = 62
                  AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                  AND DontCountHits = 0
                  AND IsRefresh = 0
                  AND URL <> ''
                GROUP BY URL
                ORDER BY PageViews DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT URL
                                       FROM hits
                                       WHERE (CounterID=62)
                                         AND (DontCountHits=0)
                                         AND (IsRefresh=0)
                                         AND (URL!='')));

query_example: SELECT URL, COUNT(*) AS PageViews
               FROM hits
               WHERE CounterID = 62
                 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                 AND DontCountHits = 0
                 AND IsRefresh = 0
                 AND URL <> ''
               GROUP BY URL
               ORDER BY PageViews DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT URL
                                    FROM hits
                                    WHERE (CounterID=62)
                                      AND (DontCountHits=0)
                                      AND (IsRefresh=0)
                                      AND (URL!='')));
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
//  Key: string_t(URL) + precomputed hash (content-based equality)
// ============================================================

struct HashedStringKey {
	string_t s;
	size_t hash;

	bool operator==(const HashedStringKey &o) const noexcept {
		if (hash != o.hash) {
			return false;
		}
		const auto a_len = s.GetSize();
		const auto b_len = o.s.GetSize();
		if (a_len != b_len) {
			return false;
		}
		if (a_len == 0) {
			return true;
		}
		return std::memcmp(s.GetDataUnsafe(), o.s.GetDataUnsafe(), a_len) == 0;
	}
};

struct HashedStringKeyHash {
	size_t operator()(const HashedStringKey &k) const noexcept {
		return k.hash;
	}
};

static inline size_t HashURL(const string_t &s) {
	return duckdb::Hash(s);
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
	absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> map;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};
	std::atomic<bool> result_emitted{false};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

// ============================================================
//  Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	StringHeap heap;
	absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> map;

	inline void AddOne(const string_t &url, const size_t h) {
		HashedStringKey probe{url, h};
		auto it = map.find(probe);
		if (it != map.end()) {
			it->second += 1;
			return;
		}
		auto copied = heap.AddString(url);
		map.emplace(HashedStringKey{copied, h}, 1);
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
	return_types.push_back(LogicalType::VARCHAR); // URL
	return_types.push_back(LogicalType::BIGINT);  // PageViews

	names.push_back("URL");
	names.push_back("PageViews");

	return make_uniq<FnBindData>();
}

// ============================================================
//  Execute (expects pre-filtered input: only URL column, already constrained)
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 1) {
		throw InvalidInputException("dbweaver expects one column: URL");
	}

	auto &v_url = input.data[0];
	if (v_url.GetType().id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("dbweaver expects URL as VARCHAR");
	}

	UnifiedVectorFormat u_url;
	v_url.ToUnifiedFormat(input.size(), u_url);

	auto &validity = u_url.validity;
	auto *data = (string_t *)u_url.data;

	constexpr idx_t BATCH = 8;

	if (validity.AllValid()) {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min<idx_t>(BATCH, input.size() - rr);

			string_t urls[BATCH];
			size_t hashes[BATCH];

			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;
				const idx_t idx = u_url.sel->get_index(ridx);
				urls[k] = data[idx];
				hashes[k] = HashURL(urls[k]);
			}
			for (idx_t k = 0; k < n; ++k) {
				l.AddOne(urls[k], hashes[k]);
			}
		}
	} else {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min<idx_t>(BATCH, input.size() - rr);
			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;
				const idx_t idx = u_url.sel->get_index(ridx);
				if (!validity.RowIsValid(idx)) {
					continue;
				}
				const string_t url = data[idx];
				const size_t h = HashURL(url);
				l.AddOne(url, h);
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Merge helper
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (auto &kv : local.map) {
		const HashedStringKey &k = kv.first;
		const int64_t cnt = kv.second;
		if (cnt <= 0) {
			continue;
		}

		auto it = g.map.find(k);
		if (it != g.map.end()) {
			it->second += cnt;
			continue;
		}

		auto copied = g.heap.AddString(k.s);
		g.map.emplace(HashedStringKey{copied, k.hash}, cnt);
	}
}

// ============================================================
//  Top10 output
// ============================================================

struct TopRow {
	int64_t c;
	string_t url;
	size_t hash;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const {
		return a.c > b.c; // min-heap by count
	}
};

template <class MapT>
static void EmitTop10FromMap(MapT &map_ref, DataChunk &out) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (auto it = map_ref.begin(); it != map_ref.end(); ++it) {
		const HashedStringKey &k = it->first;
		const int64_t cnt = it->second;
		if (cnt <= 0) {
			continue;
		}

		TopRow row;
		row.c = cnt;
		row.url = k.s;
		row.hash = k.hash;

		if (pq.size() < 10) {
			pq.push(row);
		} else if (row.c > pq.top().c) {
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
		if (a.c != b.c) {
			return a.c > b.c;
		}
		if (a.hash != b.hash) {
			return a.hash < b.hash;
		}
		const auto al = a.url.GetSize();
		const auto bl = b.url.GetSize();
		const auto ml = std::min(al, bl);
		int cmp = 0;
		if (ml > 0) {
			cmp = std::memcmp(a.url.GetDataUnsafe(), b.url.GetDataUnsafe(), ml);
		}
		if (cmp != 0) {
			return cmp < 0;
		}
		return al < bl;
	});

	auto *out_url = FlatVector::GetData<string_t>(out.data[0]);
	auto *out_pv = FlatVector::GetData<int64_t>(out.data[1]);

	idx_t out_idx = 0;
	for (idx_t i = 0; i < top.size(); ++i) {
		out_url[out_idx] = StringVector::AddString(out.data[0], top[i].url);
		out_pv[out_idx] = top[i].c;
		++out_idx;
	}
	out.SetCardinality(out_idx);
}

// ============================================================
//  Finalize
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

	const auto merged = g.merged_local_states.load(std::memory_order_acquire);
	const auto active = g.active_local_states.load(std::memory_order_acquire);

	// Only the last finisher emits result once.
	if (active > 0 && merged == active) {
		bool expected = false;
		if (g.result_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> guard(g.lock);
			EmitTop10FromMap(g.map, out);
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

void DbweaverExtension::Load(ExtensionLoader &loader) { LoadInternal(loader); }
std::string DbweaverExtension::Name() { return "dbweaver"; }
std::string DbweaverExtension::Version() const { return DuckDB::LibraryVersion(); }

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) {
	duckdb::LoadInternal(loader);
}

}