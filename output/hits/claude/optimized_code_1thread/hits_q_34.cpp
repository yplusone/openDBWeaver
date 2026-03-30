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
//  wyhash implementation
// ============================================================

static inline uint64_t _wyread8(const uint8_t *p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
static inline uint64_t _wyread4(const uint8_t *p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
static inline uint64_t _wyread3(const uint8_t *p, size_t k) {
	return (((uint64_t)p[0]) << 16) | (((uint64_t)p[k >> 1]) << 8) | p[k - 1];
}
static inline void _wymum(uint64_t *A, uint64_t *B) {
	__uint128_t r = *A; r *= *B;
	*A = (uint64_t)r; *B = (uint64_t)(r >> 64);
}
static inline uint64_t _wymix(uint64_t A, uint64_t B) {
	_wymum(&A, &B);
	return A ^ B;
}
static inline uint64_t wyhash64(const void *key, size_t len, uint64_t seed) {
	const uint8_t *p = (const uint8_t *)key;
	const uint64_t _wyp0 = 0xa0761d6478bd642fULL, _wyp1 = 0xe7037ed1a0b428dbULL, _wyp2 = 0x8ebc6af09c88c6e3ULL,
	               _wyp3 = 0x589965cc75374cc3ULL, _wyp4 = 0x1d8e4e27c47d124fULL;
	seed ^= _wymix(seed ^ _wyp0, _wyp1);
	uint64_t a, b;
	if (len <= 16) {
		if (len >= 4) {
			a = (_wyread4(p) << 32) | _wyread4(p + ((len >> 3) << 2));
			b = (_wyread4(p + len - 4) << 32) | _wyread4(p + len - 4 - ((len >> 3) << 2));
		} else if (len > 0) {
			a = _wyread3(p, len);
			b = 0;
		} else
			a = b = 0;
	} else {
		size_t i = len;
		if (i > 48) {
			uint64_t see1 = seed, see2 = seed;
			do {
				seed = _wymix(_wyread8(p) ^ _wyp1, _wyread8(p + 8) ^ seed);
				see1 = _wymix(_wyread8(p + 16) ^ _wyp2, _wyread8(p + 24) ^ see1);
				see2 = _wymix(_wyread8(p + 32) ^ _wyp3, _wyread8(p + 40) ^ see2);
				p += 48;
				i -= 48;
			} while (i > 48);
			seed ^= see1 ^ see2;
		}
		while (i > 16) {
			seed = _wymix(_wyread8(p) ^ _wyp1, _wyread8(p + 8) ^ seed);
			p += 16;
			i -= 16;
		}
		a = _wyread8(p + i - 16);
		b = _wyread8(p + i - 8);
	}
	return _wymix(_wyp4 ^ len, _wymix(a ^ _wyp1, b ^ seed));
}

// ============================================================
//  Key: string_t(URL) + precomputed hash (content-based equality)
// ============================================================

struct HashedStringKey {
	string_t s;
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
	return (size_t)wyhash64(s.GetDataUnsafe(), (size_t)s.GetSize(), 0);
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
	absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> map;

	inline void AddOne(const string_t &url, const size_t h) {
		HashedStringKey probe{url, h};
		auto it = map.find(probe);
		if (it != map.end()) {
			it->second += 1;
			return;
		}
		// Copy into local heap to ensure key memory is stable across chunks.
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

	// Expect 1 column: URL
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

	// Batch a bit to reduce loop overhead.
	constexpr idx_t BATCH = 8;

	if (validity.AllValid()) {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min(BATCH, input.size() - rr);

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
			const idx_t n = std::min(BATCH, input.size() - rr);

			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;
				const idx_t idx = u_url.sel->get_index(ridx);
				if (!validity.RowIsValid(idx)) continue;

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
	// g.lock must be held by caller
	for (auto &kv : local.map) {
		const HashedStringKey &k = kv.first;
		const int64_t cnt = kv.second;
		if (cnt <= 0) continue;

		auto it = g.map.find(k);
		if (it != g.map.end()) {
			it->second += cnt;
			continue;
		}

		// Need to copy key into global heap, since local heap will be freed.
		auto copied = g.heap.AddString(k.s);
		g.map.emplace(HashedStringKey{copied, k.hash}, cnt);
	}
}

// ============================================================
//  Finalize: adopt-first finisher + merge rest + Top10 by c
// ============================================================

struct TopRow {
	int64_t c;
	string_t url;
	size_t hash;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.c > b.c; } // min-heap by c
};
static void EmitTop10(DataChunk &out, absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> &map_ref) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash>::iterator it = map_ref.begin();
	     it != map_ref.end(); ++it) {
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
		// stable-ish tie break: hash then lexicographic
		if (a.hash != b.hash) {
			return a.hash < b.hash;
		}
		const idx_t al = a.url.GetSize();
		const idx_t bl = b.url.GetSize();
		const idx_t ml = std::min(al, bl);
		int cmp = 0;
		if (ml > 0) {
			cmp = std::memcmp(a.url.GetDataUnsafe(), b.url.GetDataUnsafe(), ml);
		}
		if (cmp != 0) {
			return cmp < 0;
		}
		return al < bl;
	});

	Vector &out_url = out.data[0];
	int64_t *out_c = FlatVector::GetData<int64_t>(out.data[1]);

	idx_t out_idx = 0;
	for (idx_t i = 0; i < top.size(); i++) {
		const TopRow &r = top[i];
		FlatVector::GetData<string_t>(out_url)[out_idx] = r.url;
		out_c[out_idx] = r.c;
		out_idx++;
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

	if (!l.merged) {
		uint8_t expected = 0;
		if (g.adopt_stage.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> guard(g.lock);

			if (g.map.empty()) {
				// Fast path: just merge into empty global map.
				MergeLocalIntoGlobal(l, g);
			} else {
				MergeLocalIntoGlobal(l, g);
			}

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