/*
query_template:
SELECT
  REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\\1') AS k,
  AVG(STRLEN(Referer)) AS l,
  COUNT(*) AS c,
  MIN(Referer) AS min_referer
FROM hits
WHERE Referer <> ''
GROUP BY k
HAVING COUNT(*) > 100000
ORDER BY l DESC
LIMIT 25;

split_template:
SELECT k, l, c, min_referer
FROM dbweaver((
  SELECT Referer
  FROM hits
));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"

#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include "absl/container/flat_hash_map.h"
#include <vector>
#include <algorithm>
#include <cstdint>
#include <functional>


// DuckDB vendored RE2
#include "re2/re2.h"
using duckdb_re2::RE2;
using duckdb_re2::StringPiece;

namespace duckdb {

// ============================================================
// Extraction logic with manual scan fast-path
// ============================================================

// Regex Pattern: '^https?://(?:www\.)?([^/]+)/.*$'
// Manual implementation must match RE2::FullMatch behavior.
static inline std::string ExtractK(const string_t &ref) {
	const char *s = ref.GetData();
	const idx_t n = ref.GetSize();

	// Shortest possible match is "http://a/" (9 bytes)
	if (n < 9) return ref.GetString();

	idx_t pos = 0;
	if (memcmp(s, "http://", 7) == 0) {
		pos = 7;
	} else if (n >= 8 && memcmp(s, "https://", 8) == 0) {
		pos = 8;
	} else {
		return ref.GetString();
	}

	idx_t domain_start = pos;
	// Optional (?:www\.)?
	if (n - pos >= 4 && memcmp(s + pos, "www.", 4) == 0) {
		domain_start = pos + 4;
	}

	const char *start_search = s + domain_start;
	const char *end = s + n;
	const char *slash = (const char *)memchr(start_search, '/', (size_t)(end - start_search));

	// Regex ([^/]+)/ requires at least one non-slash character followed by a slash
	if (!slash || slash == start_search) {
		return ref.GetString();
	}

	// RE2's .* (without dot_nl) does not match \n. 
	// FullMatch requires the whole string to match. If there is a \n after the slash, match fails.
	for (const char *p = slash; p < end; ++p) {
		if (*p == '\n') return ref.GetString();
	}

	return std::string(start_search, (size_t)(slash - start_search));
}

// ============================================================
// State Structures
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};
struct AggState {

	int64_t count = 0;
	int64_t sum_len = 0;
	std::string min_ref;
	bool initialized = false;
};
struct SortRow {
	std::string key;
	AggState state;
	double l;
};

struct SortRowComparator {
	bool operator()(const SortRow &a, const SortRow &b) const {
		if (a.l != b.l) return a.l > b.l;
		return a.key < b.key;
	}
};
struct FnGlobalState : public GlobalTableFunctionState {
	static constexpr idx_t PARTITION_COUNT = 64;
	struct AggPartition {
		std::mutex lock;
		absl::flat_hash_map<std::string, AggState> agg_map;
	};
	AggPartition partitions[PARTITION_COUNT];

	std::atomic<idx_t> remaining_local_states{0};
	std::vector<SortRow> sorted_rows;

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	absl::flat_hash_map<std::string, AggState> agg_map;
};


// ============================================================
// Table Function Implementation
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.remaining_local_states.fetch_add(1, std::memory_order_relaxed);
	return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	names.push_back("k");
	names.push_back("l");
	names.push_back("c");
	names.push_back("min_referer");

	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::DOUBLE);
	return_types.push_back(LogicalType::BIGINT);
	return_types.push_back(LogicalType::VARCHAR);

	return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	UnifiedVectorFormat ref_uvf;
	input.data[0].ToUnifiedFormat(input.size(), ref_uvf);

	const idx_t nrows = input.size();
	string_t *ref_ptr = (string_t *)ref_uvf.data;

	for (idx_t row = 0; row < nrows; ++row) {
		const idx_t i = ref_uvf.sel->get_index(row);
		if (!ref_uvf.validity.RowIsValid(i)) continue;

		const string_t vref = ref_ptr[i];
		const idx_t ref_len = vref.GetSize();
		if (ref_len == 0) continue;
		auto &agg = l.agg_map[ExtractK(vref)];

		if (!agg.initialized) {
			agg.count = 1;
			agg.sum_len = (int64_t)ref_len;
			agg.min_ref = vref.GetString();
			agg.initialized = true;
		} else {
			agg.count++;
			agg.sum_len += (int64_t)ref_len;
			const std::string cur = vref.GetString();
			if (cur < agg.min_ref) agg.min_ref = cur;
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	if (!l.merged) {
		for (const auto &entry : l.agg_map) {
			size_t h = std::hash<std::string>{}(entry.first);
			size_t part_idx = h % FnGlobalState::PARTITION_COUNT;
			auto &part = g.partitions[part_idx];
			std::lock_guard<std::mutex> guard(part.lock);
			auto &gst = part.agg_map[entry.first];
			if (!gst.initialized) {
				gst = entry.second;
			} else {
				gst.count += entry.second.count;
				gst.sum_len += entry.second.sum_len;
				if (entry.second.min_ref < gst.min_ref) gst.min_ref = entry.second.min_ref;
			}
		}
		l.merged = true;
	}

	const idx_t prev = g.remaining_local_states.fetch_sub(1, std::memory_order_acq_rel);
	if (prev == 1) {
		for (size_t i = 0; i < FnGlobalState::PARTITION_COUNT; ++i) {
			auto &part = g.partitions[i];
			for (const auto &entry : part.agg_map) {
				if (entry.second.count > 100000) {
					double avg = (double)entry.second.sum_len / (double)entry.second.count;
					g.sorted_rows.push_back({entry.first, entry.second, avg});
				}
			}
		}
		std::sort(g.sorted_rows.begin(), g.sorted_rows.end(), SortRowComparator{});

		idx_t out_idx = 0;
		const size_t max_out = std::min<size_t>(25, g.sorted_rows.size());
		for (size_t i = 0; i < max_out; ++i) {
			const auto &row = g.sorted_rows[i];
			out.SetValue(0, out_idx, Value(row.key));

			out.SetValue(1, out_idx, Value(row.l));
			out.SetValue(2, out_idx, Value::BIGINT(row.state.count));
			out.SetValue(3, out_idx, Value(row.state.min_ref));
			out_idx++;
		}
		out.SetCardinality(out_idx);
	} else {
		out.SetCardinality(0);
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
