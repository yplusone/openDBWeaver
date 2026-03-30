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

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <algorithm>

namespace duckdb {

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

	absl::flat_hash_map<int64_t, int64_t> map; // ClientIP -> count

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
	absl::flat_hash_map<int64_t, int64_t> map;

	inline void AddOne(int64_t client_ip) {
		auto it = map.find(client_ip);
		if (it != map.end()) {
			it->second += 1;
			return;
		}
		map.emplace(client_ip, 1);
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
//  Small helpers for "accept any integral type" reads
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

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();

	// Expect 1 column: ClientIP
	if (input.ColumnCount() < 1) {
		throw InvalidInputException("dbweaver expects one column: ClientIP");
	}

	auto &v_ip = input.data[0];
	const auto t_ip = v_ip.GetType().id();
	if (!IsIntegral(t_ip)) {
		throw InvalidInputException("dbweaver expects ClientIP as an integral type");
	}

	UnifiedVectorFormat u_ip;
	v_ip.ToUnifiedFormat(input.size(), u_ip);

	auto &validity = u_ip.validity;
	constexpr idx_t BATCH = 8;

	if (validity.AllValid()) {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min(BATCH, input.size() - rr);

			int64_t ips[BATCH];
			for (idx_t k = 0; k < n; ++k) {
				ips[k] = ReadInt64At(u_ip, rr + k, t_ip);
			}
			for (idx_t k = 0; k < n; ++k) {
				l.AddOne(ips[k]);
			}
		}
	} else {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min(BATCH, input.size() - rr);

			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;
				const idx_t idx = u_ip.sel->get_index(ridx);
				if (!validity.RowIsValid(idx)) continue;
				const int64_t ip = ReadInt64At(u_ip, ridx, t_ip);
				l.AddOne(ip);
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
		const int64_t ip = kv.first;
		const int64_t cnt = kv.second;
		if (cnt <= 0) continue;

		auto it = g.map.find(ip);
		if (it != g.map.end()) {
			it->second += cnt;
		} else {
			g.map.emplace(ip, cnt);
		}
	}
}

// ============================================================
//  Finalize: adopt-first finisher + merge rest + Top10 by c
// ============================================================

struct TopRow {
	int64_t c;
	int64_t client_ip;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.c > b.c; } // min-heap by c
};
static void EmitTop10(DataChunk &out, absl::flat_hash_map<int64_t, int64_t> &map_ref) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (absl::flat_hash_map<int64_t, int64_t>::iterator it = map_ref.begin(); it != map_ref.end(); ++it) {
		const int64_t ip = it->first;
		const int64_t cnt = it->second;
		if (cnt <= 0) {
			continue;
		}

		TopRow row;
		row.c = cnt;
		row.client_ip = ip;
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
		return a.client_ip < b.client_ip;
	});

	auto *out_ip = FlatVector::GetData<int64_t>(out.data[0]);
	auto *out_ip_m1 = FlatVector::GetData<int64_t>(out.data[1]);
	auto *out_ip_m2 = FlatVector::GetData<int64_t>(out.data[2]);
	auto *out_ip_m3 = FlatVector::GetData<int64_t>(out.data[3]);
	auto *out_c = FlatVector::GetData<int64_t>(out.data[4]);

	idx_t out_idx = 0;
	for (idx_t i = 0; i < top.size(); i++) {
		const TopRow &r = top[i];
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

			if (g.map.empty()) {
				g.map = std::move(l.map);
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