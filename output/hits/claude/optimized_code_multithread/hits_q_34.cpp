/*
query_template: SELECT URL, COUNT(*) AS c
                FROM hits
                GROUP BY URL
                ORDER BY c DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT URL FROM hits));

query_example: SELECT URL, COUNT(*) AS c
               FROM hits
               GROUP BY URL
               ORDER BY c DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT URL FROM hits));
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
	mutable string_t s;
	size_t hash;


	bool operator==(const HashedStringKey &o) const noexcept {
		if (hash != o.hash) return false;
		const auto a_len = s.GetSize();
		const auto b_len = o.s.GetSize();
		if (a_len != b_len) return false;
		if (a_len == 0) return true;
		const char *a = s.GetDataUnsafe();
		const char *b = o.s.GetDataUnsafe();
		return std::memcmp(a, b, a_len) == 0;
	}
};

struct HashedStringKeyHash {
	size_t operator()(const HashedStringKey &k) const noexcept { return k.hash; }
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
	static constexpr size_t NUM_PARTITIONS = 128;

	struct Partition {
		std::mutex lock;
		StringHeap heap;
		absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> map;
	};

	Partition partitions[NUM_PARTITIONS];

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};

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
		auto res = map.try_emplace(probe, 1);
		if (res.second) {
			// New entry: copy into local heap to ensure key memory is stable.
			res.first->first.s = heap.AddString(url);
		} else {
			res.first->second += 1;
		}
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
	return_types.push_back(LogicalType::BIGINT);  // c

	names.push_back("URL");
	names.push_back("c");

	return make_uniq<FnBindData>();
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;
	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 1) {
		throw InvalidInputException("dbweaver expects one column: URL");
	}
	auto &v_url = input.data[0];
	if (v_url.GetType().id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("dbweaver expects URL as VARCHAR");
	}
	Vector hashes(LogicalType::HASH);
	VectorOperations::Hash(v_url, hashes, input.size());
	auto hash_data = FlatVector::GetData<hash_t>(hashes);

	UnifiedVectorFormat u_url;
	v_url.ToUnifiedFormat(input.size(), u_url);
	const auto *data = (const string_t *)u_url.data;
	const auto &validity = u_url.validity;

	for (idx_t i = 0; i < input.size(); i++) {
		const idx_t idx = u_url.sel->get_index(i);
		if (!validity.RowIsValid(idx)) continue;
		const string_t url = data[idx];
		l.AddOne(url, hash_data[i]);
	}


	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize Helpers
// ============================================================

struct TopRow {
	int64_t c;
	string_t url;
	size_t hash;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.c > b.c; }
};

static void MergeLocalIntoGlobal(FnLocalState &l, FnGlobalState &g) {
	for (auto &kv : l.map) {
		const HashedStringKey &k = kv.first;
		const int64_t cnt = kv.second;
		if (cnt <= 0) continue;

		idx_t part_idx = k.hash % FnGlobalState::NUM_PARTITIONS;
		FnGlobalState::Partition &part = g.partitions[part_idx];
		std::lock_guard<std::mutex> guard(part.lock);
		auto res = part.map.try_emplace(k, cnt);
		if (res.second) {
			res.first->first.s = part.heap.AddString(k.s);
		} else {
			res.first->second += cnt;
		}

	}
}

static void FinalizeTop10(DataChunk &out, std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq) {
	std::vector<TopRow> top;
	top.reserve(pq.size());
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}

	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.c != b.c) return a.c > b.c;
		if (a.hash != b.hash) return a.hash < b.hash;
		const idx_t al = a.url.GetSize();
		const idx_t bl = b.url.GetSize();
		const idx_t ml = std::min(al, bl);
		int cmp = 0;
		if (ml > 0) {
			cmp = std::memcmp(a.url.GetDataUnsafe(), b.url.GetDataUnsafe(), ml);
		}
		if (cmp != 0) return cmp < 0;
		return al < bl;
	});

	Vector &out_url = out.data[0];
	int64_t *out_c = FlatVector::GetData<int64_t>(out.data[1]);
	idx_t out_idx = 0;
	for (const auto &r : top) {
		FlatVector::GetData<string_t>(out_url)[out_idx] = r.url;
		out_c[out_idx] = r.c;
		out_idx++;
	}
	out.SetCardinality(out_idx);
}

static void EmitTop10Local(DataChunk &out, absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> &map) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	for (auto &kv : map) {
		TopRow row{kv.second, kv.first.s, kv.first.hash};
		if (pq.size() < 10) {
			pq.push(row);
		} else if (row.c > pq.top().c) {
			pq.pop();
			pq.push(row);
		}
	}
	FinalizeTop10(out, pq);
}

static void EmitTop10Global(DataChunk &out, FnGlobalState &g) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	for (size_t i = 0; i < FnGlobalState::NUM_PARTITIONS; ++i) {
		for (auto &kv : g.partitions[i].map) {
			TopRow row{kv.second, kv.first.s, kv.first.hash};
			if (pq.size() < 10) {
				pq.push(row);
			} else if (row.c > pq.top().c) {
				pq.pop();
				pq.push(row);
			}
		}
	}
	FinalizeTop10(out, pq);
}

// ============================================================
//  Finalize: Parallel Merge into Partitioned Global State
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);
	if (active <= 1) {
		EmitTop10Local(out, l.map);
		return OperatorFinalizeResultType::FINISHED;
	}

	if (!l.merged) {
		MergeLocalIntoGlobal(l, g);
		l.merged = true;
		if (g.merged_local_states.fetch_add(1, std::memory_order_acq_rel) == active - 1) {
			EmitTop10Global(out, g);
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