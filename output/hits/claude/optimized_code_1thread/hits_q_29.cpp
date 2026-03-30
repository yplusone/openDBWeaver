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
#include <vector>
#include <algorithm>
#include <cstdint>

// Abseil for flat_hash_map and heterogeneous lookup
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"

namespace duckdb {

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

struct AggState {
	int64_t count = 0;
	int64_t sum_len = 0;
	std::string min_ref;
};

// Heterogeneous lookup support for absl::flat_hash_map
struct StringHash {
	using is_transparent = void;
	size_t operator()(absl::string_view sv) const {
		return absl::Hash<absl::string_view>{}(sv);
	}
	size_t operator()(const std::string &s) const {
		return absl::Hash<std::string>{}(s);
	}
};

struct StringEq {
	using is_transparent = void;
	bool operator()(absl::string_view a, absl::string_view b) const {
		return a == b;
	}
};

using AggMap = absl::flat_hash_map<std::string, AggState, StringHash, StringEq>;

struct SortRow {
	std::string key;
	AggState state;
	double l;
};

struct SortRowComparator {
	bool operator()(const SortRow &a, const SortRow &b) const {
		return a.l > b.l;
	}
};

struct SortState {
	std::vector<SortRow> rows;
	bool sorted = false;

	inline void AddRow(const std::string &key, const AggState &state) {
		const double avg = state.count ? (double)state.sum_len / (double)state.count : 0.0;
		rows.push_back(SortRow{key, state, avg});
	}

	inline void SortNow() {
		if (!sorted) {
			std::sort(rows.begin(), rows.end(), SortRowComparator{});
			sorted = true;
		}
	}
};

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;
	std::atomic<idx_t> remaining_local_states{0};
	AggMap agg_map;
	SortState sort_state;

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	AggMap agg_map;
};

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

	string_t *ref_ptr = (string_t *)ref_uvf.data;
	const idx_t nrows = input.size();

	for (idx_t row = 0; row < nrows; ++row) {
		const idx_t i = ref_uvf.sel->get_index(row);
		if (!ref_uvf.validity.RowIsValid(i)) continue;

		const string_t vref = ref_ptr[i];
		const idx_t ref_len = vref.GetSize();
		if (ref_len == 0) continue;

		const char* data = vref.GetData();
		const size_t rlen = (size_t)ref_len;
		absl::string_view k_view;
		bool matched = false;

		// Hand-written URL extractor logic replacing regex '^https?://(?:www\.)?([^/]+)/.*$'
		size_t pos = 0;
		if (rlen >= 7 && memcmp(data, "http://", 7) == 0) {
			pos = 7;
		} else if (rlen >= 8 && memcmp(data, "https://", 8) == 0) {
			pos = 8;
		}

		if (pos > 0) {
			size_t domain_pos = pos;
			if (rlen - domain_pos >= 4 && memcmp(data + domain_pos, "www.", 4) == 0) {
				domain_pos += 4;
			}
			const char* domain_start = data + domain_pos;
			const char* slash_ptr = (const char*)memchr(domain_start, '/', rlen - domain_pos);
			// Regex ^...([^/]+)/.*$ requires at least one char in capture group and a following slash
			if (slash_ptr && slash_ptr > domain_start) {
				// RE2's .* by default does not match newlines. If a newline exists after the first slash,
				// FullMatch with $ will fail, returning the original string.
				const char* remainder = slash_ptr;
				const size_t remainder_len = rlen - (size_t)(remainder - data);
				if (!memchr(remainder, '\n', remainder_len)) {
					k_view = absl::string_view(domain_start, (size_t)(slash_ptr - domain_start));
					matched = true;
				}
			}
		}

		if (!matched) {
			k_view = absl::string_view(data, rlen);
		}

		auto it = l.agg_map.find(k_view);
		if (it == l.agg_map.end()) {
			AggState state;
			state.count = 1;
			state.sum_len = (int64_t)ref_len;
			state.min_ref.assign(data, rlen);
			l.agg_map.emplace(std::string(k_view), std::move(state));
		} else {
			auto &agg = it->second;
			agg.count++;
			agg.sum_len += (int64_t)ref_len;
			absl::string_view sv(data, rlen);
			if (sv < agg.min_ref) {
				agg.min_ref.assign(data, rlen);
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	if (!l.merged) {
		std::lock_guard<std::mutex> guard(g.lock);
		for (auto &entry : l.agg_map) {
			auto it = g.agg_map.find(entry.first);
			if (it == g.agg_map.end()) {
				g.agg_map.emplace(std::move(entry.first), std::move(entry.second));
			} else {
				auto &gst = it->second;
				gst.count += entry.second.count;
				gst.sum_len += entry.second.sum_len;
				if (entry.second.min_ref < gst.min_ref) {
					gst.min_ref = std::move(entry.second.min_ref);
				}
			}
		}
		l.merged = true;
	}

	const idx_t prev = g.remaining_local_states.fetch_sub(1, std::memory_order_acq_rel);
	if (prev == 1) {
		std::lock_guard<std::mutex> guard(g.lock);

		for (const auto &entry : g.agg_map) {
			if (entry.second.count > 100000) {
				g.sort_state.AddRow(entry.first, entry.second);
			}
		}

		g.sort_state.SortNow();

		idx_t out_idx = 0;
		const size_t max_out = std::min<size_t>(25, g.sort_state.rows.size());
		for (size_t i = 0; i < max_out; ++i) {
			const auto &row = g.sort_state.rows[i];
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
