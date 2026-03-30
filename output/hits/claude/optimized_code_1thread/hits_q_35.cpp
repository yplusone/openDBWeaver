/*
query_template: SELECT 1, URL, COUNT(*) AS c
                FROM hits
                GROUP BY 1, URL
                ORDER BY c DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT URL FROM hits));

query_example: SELECT 1, URL, COUNT(*) AS c
               FROM hits
               GROUP BY 1, URL
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
//  Simple Bump Allocator (Slab Allocator)
// ============================================================

struct SimpleBumpAllocator {
	std::vector<std::unique_ptr<char[]>> chunks;
	char *ptr = nullptr;
	char *end = nullptr;
	static constexpr size_t CHUNK_SIZE = 256 * 1024 * 1024; // 256MB slabs as recommended

	SimpleBumpAllocator() = default;
	SimpleBumpAllocator(const SimpleBumpAllocator &) = delete;
	SimpleBumpAllocator &operator=(const SimpleBumpAllocator &) = delete;

	inline string_t AddString(const string_t &s) {
		const uint32_t len = s.GetSize();
		if (len <= 12) { // string_t::INLINE_LENGTH
			return s;
		}
		if (ptr + len > end) {
			AllocateNewChunk(len);
		}
		char *dest = ptr;
		std::memcpy(dest, s.GetDataUnsafe(), len);
		ptr += len;
		return string_t(dest, len);
	}

private:
	void AllocateNewChunk(uint32_t min_len) {
		size_t alloc_size = std::max(CHUNK_SIZE, (size_t)min_len);
		auto chunk = std::unique_ptr<char[]>(new char[alloc_size]);
		ptr = chunk.get();
		end = ptr + alloc_size;
		chunks.push_back(std::move(chunk));
	}
};

// ============================================================
//  Stored row: URL + count
// ============================================================

struct DbweaverEntry {
	string_t url;
	int64_t count;
};
static inline uint64_t HashURL(const string_t &s) {
	const char *data = s.GetDataUnsafe();
	uint64_t len = s.GetSize();
	const uint64_t _wyp0 = 0xa0761d6478bd642fULL, _wyp1 = 0xe7037ed1a0b428dbULL,
	               _wyp2 = 0x8ebc6af09c88c6e3ULL, _wyp3 = 0x589965cc75374cc3ULL;
	auto wymum = [](uint64_t A, uint64_t B) -> uint64_t {
		__uint128_t r = A; r *= B;
		return (uint64_t)r ^ (uint64_t)(r >> 64);
	};
	auto wyr8 = [](const uint8_t *p) -> uint64_t {
		uint64_t v; std::memcpy(&v, p, 8); return v;
	};
	auto wyr4 = [](const uint8_t *p) -> uint64_t {
		uint32_t v; std::memcpy(&v, p, 4); return v;
	};

	uint64_t seed = _wyp0;
	const uint8_t *p = (const uint8_t *)data;
	uint64_t a, b;
	if (len <= 16) {
		if (len >= 8) {
			a = wyr8(p); b = wyr8(p + len - 8);
		} else if (len >= 4) {
			a = wyr4(p); b = wyr4(p + len - 4);
		} else if (len > 0) {
			a = ((uint64_t)p[0] << 16) | ((uint64_t)p[len >> 1] << 8) | p[len - 1];
			b = 0;
		} else { a = b = 0; }
	} else {
		const uint8_t *p2 = p;
		uint64_t i = len;
		if (i > 48) {
			uint64_t see1 = seed, see2 = seed;
			do {
				seed = wymum(wyr8(p2) ^ _wyp1, wyr8(p2 + 8) ^ seed);
				see1 = wymum(wyr8(p2 + 16) ^ _wyp2, wyr8(p2 + 24) ^ see1);
				see2 = wymum(wyr8(p2 + 32) ^ _wyp3, wyr8(p2 + 40) ^ see2);
				p2 += 48; i -= 48;
			} while (i > 48);
			seed ^= see1 ^ see2;
		}
		while (i > 16) {
			seed = wymum(wyr8(p2) ^ _wyp1, wyr8(p2 + 8) ^ seed);
			i -= 16; p2 += 16;
		}
		a = wyr8(p2 + i - 16); b = wyr8(p2 + i - 8);
	}
	return wymum(_wyp2 ^ len, wymum(a ^ _wyp3, b ^ seed));
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

	SimpleBumpAllocator heap;
	absl::flat_hash_map<uint64_t, DbweaverEntry> map;

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

	SimpleBumpAllocator heap;
	absl::flat_hash_map<uint64_t, DbweaverEntry> map;

	inline void AddOne(const string_t &url, const uint64_t h) {
		auto it = map.find(h);
		if (it != map.end()) {
			it->second.count += 1;
			return;
		}
		auto copied = heap.AddString(url);
		map.emplace(h, DbweaverEntry{copied, 1});
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
	return_types.push_back(LogicalType::INTEGER); // 1
	return_types.push_back(LogicalType::VARCHAR); // URL
	return_types.push_back(LogicalType::BIGINT);  // c

	names.push_back("1");
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
	const bool is_flat = (v_url.GetVectorType() == VectorType::FLAT_VECTOR);

	constexpr idx_t BATCH = 8;
	if (validity.AllValid()) {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min(BATCH, input.size() - rr);

			string_t urls[BATCH];
			uint64_t hashes[BATCH];

			if (is_flat) {
				for (idx_t k = 0; k < n; ++k) {
					const idx_t idx = rr + k;
					urls[k] = data[idx];
					hashes[k] = HashURL(urls[k]);
				}
			} else {
				for (idx_t k = 0; k < n; ++k) {
					const idx_t idx = u_url.sel->get_index(rr + k);
					urls[k] = data[idx];
					hashes[k] = HashURL(urls[k]);
				}
			}

			for (idx_t k = 0; k < n; ++k) {
				l.map.prefetch(hashes[k]);
			}

			for (idx_t k = 0; k < n; ++k) {
				l.AddOne(urls[k], hashes[k]);
			}
		}
	} else {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min(BATCH, input.size() - rr);

			string_t urls[BATCH];
			uint64_t hashes[BATCH];
			idx_t valid_count = 0;

			if (is_flat) {
				for (idx_t k = 0; k < n; ++k) {
					const idx_t idx = rr + k;
					if (!validity.RowIsValid(idx)) continue;
					urls[valid_count] = data[idx];
					hashes[valid_count] = HashURL(urls[valid_count]);
					valid_count++;
				}
			} else {
				for (idx_t k = 0; k < n; ++k) {
					const idx_t idx = u_url.sel->get_index(rr + k);
					if (!validity.RowIsValid(idx)) continue;
					urls[valid_count] = data[idx];
					hashes[valid_count] = HashURL(urls[valid_count]);
					valid_count++;
				}
			}

			for (idx_t k = 0; k < valid_count; ++k) {
				l.map.prefetch(hashes[k]);
			}

			for (idx_t k = 0; k < valid_count; ++k) {
				l.AddOne(urls[k], hashes[k]);
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
		const uint64_t h = kv.first;
		const DbweaverEntry &val = kv.second;
		if (val.count <= 0) continue;

		auto it = g.map.find(h);
		if (it != g.map.end()) {
			it->second.count += val.count;
			continue;
		}

		auto copied = g.heap.AddString(val.url);
		g.map.emplace(h, DbweaverEntry{copied, val.count});
	}
}

// ============================================================
//  Finalize: adopt-first finisher + merge rest + Top10 by c
// ============================================================

struct TopRow {
	int64_t c;
	string_t url;
	uint64_t hash;
};
struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.c > b.c; } // min-heap by c
};

template <class MapT>
static void EmitTop10(DataChunk &out, MapT &map_ref) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (auto &kv : map_ref) {
		const uint64_t h = kv.first;
		const auto &entry = kv.second;
		if (entry.count <= 0) continue;

		TopRow row{entry.count, entry.url, h};
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
		if (a.c != b.c) return a.c > b.c;
		if (a.hash != b.hash) return a.hash < b.hash;
		return a.url < b.url;
	});

	auto *out_one = FlatVector::GetData<int32_t>(out.data[0]);
	auto *out_url = FlatVector::GetData<string_t>(out.data[1]);
	auto *out_c = FlatVector::GetData<int64_t>(out.data[2]);

	idx_t out_idx = 0;
	for (size_t i = 0; i < top.size(); ++i) {
		const TopRow &r = top[i];
		out_one[out_idx] = 1;
		out_url[out_idx] = r.url;
		out_c[out_idx] = r.c;
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

	if (!l.merged) {
		uint8_t expected = 0;
		if (g.adopt_stage.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> guard(g.lock);

			// Always merge using the helper
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