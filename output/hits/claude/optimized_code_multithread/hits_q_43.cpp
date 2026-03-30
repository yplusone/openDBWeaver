/*
query_template: SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;

split_template: select * from dbweaver((SELECT EventTime FROM hits WHERE (CounterID=62) AND (EventDate>='2013-07-14'::DATE AND EventDate<='2013-07-15'::DATE) AND (IsRefresh=0) AND (DontCountHits=0)));
query_example: SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;

split_query: select * from dbweaver((SELECT EventTime FROM hits WHERE (CounterID=62) AND (EventDate>='2013-07-14'::DATE AND EventDate<='2013-07-15'::DATE) AND (IsRefresh=0) AND (DontCountHits=0)));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"

#include <atomic>
#include <limits>
#include <mutex>
#include <algorithm>
#include <vector>
#include <cstring>

namespace duckdb {

// Constants for the 2-day window (2013-07-14 to 2013-07-15)
// 2013-07-14 00:00:00 is 1,373,760,000,000,000 microseconds since epoch
static constexpr int64_t BASE_TIMESTAMP_MICROS = 1373760000000000LL;
static constexpr int64_t MINUTE_MICROS = 60LL * 1000000LL;
static constexpr idx_t NUM_MINUTES = 2880; // 2 days * 1440 minutes/day

// ============================================================
// Bind Data
// ============================================================
struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

// ============================================================
// Global / Local State
// ============================================================
struct FnGlobalState : public GlobalTableFunctionState {
	int64_t global_agg_array[NUM_MINUTES];

	std::mutex lock;
	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};
	std::atomic<bool> output_emitted {false};

	FnGlobalState() {
		std::memset(global_agg_array, 0, sizeof(global_agg_array));
	}

	idx_t MaxThreads() const override {
		return std::numeric_limits<idx_t>::max();
	}
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
	int64_t local_agg_array[NUM_MINUTES];
	bool merged = false;

	FnLocalState() {
		std::memset(local_agg_array, 0, sizeof(local_agg_array));
	}
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                       GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);
	return make_uniq<FnLocalState>();
}

// ============================================================
// Bind
// ============================================================
static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	// Output schema: M, PageViews
	return_types.emplace_back(LogicalType::TIMESTAMP);
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("M");
	names.emplace_back("PageViews");

	return make_uniq<FnBindData>();
}

// ============================================================
// Execute (aggregate locally using flat array)
// Input: EventTime
// ============================================================
static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	UnifiedVectorFormat event_time_uvf;
	input.data[0].ToUnifiedFormat(input.size(), event_time_uvf);
	auto *event_time_ptr = reinterpret_cast<timestamp_t *>(event_time_uvf.data);

	for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
		const idx_t i_event_time = event_time_uvf.sel->get_index(row_idx);
		if (event_time_uvf.validity.RowIsValid(i_event_time)) {
			const int64_t micros = event_time_ptr[i_event_time].value;
			const int64_t offset = micros - BASE_TIMESTAMP_MICROS;
			if (offset >= 0) {
				const idx_t minute_idx = (idx_t)(offset / MINUTE_MICROS);
				if (minute_idx < NUM_MINUTES) {
					l.local_agg_array[minute_idx]++;
				}
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
// Finalize (merge local arrays and emit OFFSET/LIMIT results)
// ============================================================
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                             DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	// Merge local state exactly once per thread
	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			for (idx_t i = 0; i < NUM_MINUTES; ++i) {
				g.global_agg_array[i] += l.local_agg_array[i];
			}
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	// Only one thread emits final result after all local states have merged.
	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	bool should_emit = false;
	if (active > 0 && merged == active) {
		bool expected = false;
		should_emit = g.output_emitted.compare_exchange_strong(expected, true, std::memory_order_relaxed);
	}

	if (!should_emit) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	// Traversing the array indices already provides ORDER BY M ASC.
	// Apply OFFSET 1000 and LIMIT 10.
	const idx_t offset = 1000;
	const idx_t limit = 10;
	idx_t skipped = 0;
	idx_t produced = 0;

	for (idx_t i = 0; i < NUM_MINUTES && produced < limit; ++i) {
		if (g.global_agg_array[i] > 0) {
			if (skipped < offset) {
				skipped++;
			} else {
				const timestamp_t m_val(BASE_TIMESTAMP_MICROS + (int64_t)i * MINUTE_MICROS);
				out.SetValue(0, produced, Value::TIMESTAMP(m_val));
				out.SetValue(1, produced, Value::BIGINT(g.global_agg_array[i]));
				produced++;
			}
		}
	}

	out.SetCardinality(produced);
	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
// Extension Registration
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
