/*
query_template: SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c
                FROM hits
                GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3
                ORDER BY c DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT ClientIP FROM hits));

query_example: SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c
               FROM hits
               GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3
               ORDER BY c DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT ClientIP FROM hits));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>

namespace duckdb {

// ============================================================
//  Specialized Flat Aggregate Table
// ============================================================

struct SimpleTable {
	struct Slot {
		int64_t key;
		int64_t count;
	};

	std::vector<Slot> slots;
	size_t mask;
	size_t count_entries;

	SimpleTable(size_t initial_cap = 1024) : count_entries(0) {
		// Ensure power of two
		slots.resize(initial_cap, {0, 0});
		mask = initial_cap - 1;
	}

	static inline uint64_t Hash(int64_t key) {
		// Fast 64-bit mixer
		return uint64_t(key) * 0xbf58476d1ce4e5b9ULL;
	}

	inline void add(int64_t key, int64_t count = 1) {
		if (count_entries * 2 > slots.size()) {
			resize();
		}
		add_no_resize(key, count);
	}

	inline void add_no_resize(int64_t key, int64_t count) {
		uint64_t h = Hash(key) & mask;
		while (slots[h].count > 0 && slots[h].key != key) {
			h = (h + 1) & mask;
		}
		if (slots[h].count == 0) {
			slots[h].key = key;
			count_entries++;
		}
		slots[h].count += count;
	}

	void resize() {
		std::vector<Slot> old_slots = std::move(slots);
		size_t new_cap = old_slots.size() * 2;
		slots.assign(new_cap, {0, 0});
		mask = new_cap - 1;
		count_entries = 0;
		for (const auto &s : old_slots) {
			if (s.count > 0) {
				add_no_resize(s.key, s.count);
			}
		}
	}
};

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
	SimpleTable map;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};
	std::atomic<uint8_t> adopt_stage{0};

	FnGlobalState() : map(1 << 20) {} // Start with 1M slots

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

// ============================================================
//  Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	SimpleTable map;

	FnLocalState() : map(1 << 16) {} // Start with 64K slots
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
	return_types.push_back(LogicalType::BIGINT); // ClientIP
	return_types.push_back(LogicalType::BIGINT); // ClientIP - 1
	return_types.push_back(LogicalType::BIGINT); // ClientIP - 2
	return_types.push_back(LogicalType::BIGINT); // ClientIP - 3
	return_types.push_back(LogicalType::BIGINT); // c

	names.push_back("ClientIP");
	names.push_back("(ClientIP - 1)");
	names.push_back("(ClientIP - 2)");
	names.push_back("(ClientIP - 3)");
	names.push_back("c");

	return make_uniq<FnBindData>();
}

// ============================================================
//  Helper: Read integral type
// ============================================================

static inline bool IsIntegral(LogicalTypeId id) {
	return id == LogicalTypeId::TINYINT || id == LogicalTypeId::SMALLINT || id == LogicalTypeId::INTEGER ||
	       id == LogicalTypeId::BIGINT || id == LogicalTypeId::UTINYINT || id == LogicalTypeId::USMALLINT ||
	       id == LogicalTypeId::UINTEGER || id == LogicalTypeId::UBIGINT || id == LogicalTypeId::BOOLEAN;
}

static inline int64_t ReadInt64At(const UnifiedVectorFormat &uvf, idx_t ridx, LogicalTypeId tid) {
	const idx_t idx = uvf.sel->get_index(ridx);
	switch (tid) {
	case LogicalTypeId::BOOLEAN: return (int64_t)((bool *)uvf.data)[idx];
	case LogicalTypeId::TINYINT: return (int64_t)((int8_t *)uvf.data)[idx];
	case LogicalTypeId::SMALLINT: return (int64_t)((int16_t *)uvf.data)[idx];
	case LogicalTypeId::INTEGER: return (int64_t)((int32_t *)uvf.data)[idx];
	case LogicalTypeId::BIGINT: return (int64_t)((int64_t *)uvf.data)[idx];
	case LogicalTypeId::UTINYINT: return (int64_t)((uint8_t *)uvf.data)[idx];
	case LogicalTypeId::USMALLINT: return (int64_t)((uint16_t *)uvf.data)[idx];
	case LogicalTypeId::UINTEGER: return (int64_t)((uint32_t *)uvf.data)[idx];
	case LogicalTypeId::UBIGINT: return (int64_t)((uint64_t *)uvf.data)[idx];
	default: return 0;
	}
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();
	if (input.ColumnCount() < 1) throw InvalidInputException("dbweaver expects one column: ClientIP");

	auto &v_ip = input.data[0];
	const auto t_ip = v_ip.GetType().id();
	if (!IsIntegral(t_ip)) throw InvalidInputException("dbweaver expects ClientIP as an integral type");

	UnifiedVectorFormat u_ip;
	v_ip.ToUnifiedFormat(input.size(), u_ip);

	// Ensure enough space to avoid frequent resizing within the batch loop
	if (l.map.count_entries + input.size() > l.map.slots.size() / 2) {
		l.map.resize();
	}

	auto &validity = u_ip.validity;
	const idx_t size = input.size();

	// Optimized path for Flat (Identity) Selection Vector
	if (u_ip.sel->data() == nullptr) {
		if (t_ip == LogicalTypeId::BIGINT) {
			const int64_t *data = (const int64_t *)u_ip.data;
			if (validity.AllValid()) {
				for (idx_t i = 0; i < size; ++i) {
					if (i + 8 < size) {
						__builtin_prefetch(&l.map.slots[SimpleTable::Hash(data[i + 8]) & l.map.mask], 0, 1);
					}
					l.map.add_no_resize(data[i], 1);
				}
			} else {
				for (idx_t i = 0; i < size; ++i) {
					if (validity.RowIsValid(i)) {
						l.map.add_no_resize(data[i], 1);
					}
				}
			}
			return OperatorResultType::NEED_MORE_INPUT;
		} else if (t_ip == LogicalTypeId::INTEGER) {
			const int32_t *data = (const int32_t *)u_ip.data;
			if (validity.AllValid()) {
				for (idx_t i = 0; i < size; ++i) {
					if (i + 8 < size) {
						__builtin_prefetch(&l.map.slots[SimpleTable::Hash(data[i + 8]) & l.map.mask], 0, 1);
					}
					l.map.add_no_resize(data[i], 1);
				}
			} else {
				for (idx_t i = 0; i < size; ++i) {
					if (validity.RowIsValid(i)) {
						l.map.add_no_resize(data[i], 1);
					}
				}
			}
			return OperatorResultType::NEED_MORE_INPUT;
		}
	}

	// Fallback for non-flat or other integral types
	if (validity.AllValid()) {
		for (idx_t i = 0; i < size; ++i) {
			if (i + 8 < size) {
				const int64_t next_ip = ReadInt64At(u_ip, i + 8, t_ip);
				__builtin_prefetch(&l.map.slots[SimpleTable::Hash(next_ip) & l.map.mask], 0, 1);
			}
			const int64_t ip = ReadInt64At(u_ip, i, t_ip);
			l.map.add_no_resize(ip, 1);
		}
	} else {
		for (idx_t i = 0; i < size; ++i) {
			const idx_t idx = u_ip.sel->get_index(i);
			if (validity.RowIsValid(idx)) {
				const int64_t ip = ReadInt64At(u_ip, i, t_ip);
				l.map.add_no_resize(ip, 1);
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize Helpers
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (const auto &s : local.map.slots) {
		if (s.count > 0) {
			g.map.add(s.key, s.count);
		}
	}
}

struct TopRow {
	int64_t c;
	int64_t client_ip;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.c > b.c; }
};

static void EmitTop10(DataChunk &out, SimpleTable &map_ref) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (const auto &s : map_ref.slots) {
		if (s.count <= 0) continue;
		TopRow row = {s.count, s.key};
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
		return a.client_ip < b.client_ip;
	});

	idx_t out_idx = 0;
	auto *out_ip = FlatVector::GetData<int64_t>(out.data[0]);
	auto *out_ip_m1 = FlatVector::GetData<int64_t>(out.data[1]);
	auto *out_ip_m2 = FlatVector::GetData<int64_t>(out.data[2]);
	auto *out_ip_m3 = FlatVector::GetData<int64_t>(out.data[3]);
	auto *out_c = FlatVector::GetData<int64_t>(out.data[4]);

	for (const auto &r : top) {
		const int64_t ip = r.client_ip;
		out_ip[out_idx] = ip;
		out_ip_m1[out_idx] = ip - 1;
		out_ip_m2[out_idx] = ip - 2;
		out_ip_m3[out_idx] = ip - 3;
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
			if (g.map.count_entries == 0) {
				g.map = std::move(l.map);
			} else {
				MergeLocalIntoGlobal(l, g);
			}
			g.adopt_stage.store(2, std::memory_order_release);
			l.merged = true;
			g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
		} else {
			while (g.adopt_stage.load(std::memory_order_acquire) == 1) std::this_thread::yield();
			std::lock_guard<std::mutex> guard(g.lock);
			MergeLocalIntoGlobal(l, g);
			l.merged = true;
			g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
		}
	}

	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
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
DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) { duckdb::LoadInternal(loader); }
}
