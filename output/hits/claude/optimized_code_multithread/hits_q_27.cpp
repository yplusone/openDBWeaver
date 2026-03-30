/*
query_template: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase, EventTime FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase, EventTime FROM hits WHERE (SearchPhrase!='')));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"

#include <atomic>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

namespace duckdb {

// ============================================================
//  Helpers: maintain 10 smallest tuples by (EventTime ASC, SearchPhrase ASC)
// ============================================================

struct Entry {
	timestamp_t ts;
	std::string phrase;

	// lex ascending order
	inline bool operator<(const Entry &other) const noexcept {
		if (ts.value != other.ts.value) {
			return ts.value < other.ts.value;
		}
		return phrase < other.phrase;
	}
};

static inline int LexCompareStringT(const string_t &a, const std::string &b) noexcept {
	const auto alen = (idx_t)a.GetSize();
	const auto blen = (idx_t)b.size();
	const auto n = alen < blen ? alen : blen;
	int cmp = 0;
	if (n > 0) {
		cmp = std::memcmp(a.GetData(), b.data(), n);
	}
	if (cmp != 0) {
		return cmp;
	}
	if (alen < blen)
		return -1;
	if (alen > blen)
		return 1;
	return 0;
}

// ============================================================
//  Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

// ============================================================
//  Global / Local state
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;
	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};

	std::vector<Entry> topk;

	FnGlobalState() {
		topk.reserve(10);
	}

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	std::vector<Entry> topk;

	FnLocalState() {
		topk.reserve(10);
	}
};

// ============================================================
//  Init / Bind
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);
	return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("SearchPhrase");
	return make_uniq<FnBindData>();
}

// ============================================================
//  Execute: consume input chunks, maintain local top-10 smallest
// ============================================================
static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                   DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	D_ASSERT(input.ColumnCount() == 2);
	auto &phrase_vec = input.data[0];
	auto &time_vec = input.data[1];
	const auto count = input.size();

	int64_t max_ts = std::numeric_limits<int64_t>::max();
	bool full = (l.topk.size() >= 10);
	if (full) {
		max_ts = l.topk.back().ts.value;
	}

	if (phrase_vec.GetVectorType() == VectorType::FLAT_VECTOR && time_vec.GetVectorType() == VectorType::FLAT_VECTOR) {
		// Fast path for Flat vectors: avoids selection vector overhead and UnifiedVectorFormat construction
		auto phrase_data = FlatVector::GetData<string_t>(phrase_vec);
		auto time_data = FlatVector::GetData<timestamp_t>(time_vec);
		auto &phrase_validity = FlatVector::Validity(phrase_vec);
		auto &time_validity = FlatVector::Validity(time_vec);

		for (idx_t i = 0; i < count; i++) {
			if (!time_validity.RowIsValid(i)) continue;
			const auto ts_val = time_data[i].value;
			if (full && ts_val > max_ts) continue;

			if (!phrase_validity.RowIsValid(i)) continue;
			const auto &s = phrase_data[i];
			const auto s_size = s.GetSize();
			if (s_size == 0) continue;

			if (!full) {
				l.topk.push_back({timestamp_t(ts_val), std::string(s.GetData(), s_size)});
				std::sort(l.topk.begin(), l.topk.end());
				if (l.topk.size() == 10) {
					full = true;
					max_ts = l.topk.back().ts.value;
				}
			} else {
				if (ts_val < max_ts || (ts_val == max_ts && LexCompareStringT(s, l.topk.back().phrase) < 0)) {
					l.topk.back().ts.value = ts_val;
					l.topk.back().phrase.assign(s.GetData(), s_size);
					for (int j = 8; j >= 0; --j) {
						if (l.topk[j + 1] < l.topk[j]) {
							std::swap(l.topk[j], l.topk[j + 1]);
						} else {
							break;
						}
					}
					max_ts = l.topk.back().ts.value;
				}
			}
		}
	} else {
		// Fallback to UnifiedVectorFormat for non-flat vectors
		UnifiedVectorFormat phrase_uf;
		UnifiedVectorFormat time_uf;
		phrase_vec.ToUnifiedFormat(count, phrase_uf);
		time_vec.ToUnifiedFormat(count, time_uf);

		auto phrase_data = (const string_t *)phrase_uf.data;
		auto time_data = (const timestamp_t *)time_uf.data;

		for (idx_t i = 0; i < count; i++) {
			const auto tr = time_uf.sel->get_index(i);
			if (!time_uf.validity.RowIsValid(tr)) {
				continue;
			}
			const auto ts_val = time_data[tr].value;
			if (full && ts_val > max_ts) {
				continue;
			}

			const auto pr = phrase_uf.sel->get_index(i);
			if (!phrase_uf.validity.RowIsValid(pr)) {
				continue;
			}
			const auto &s = phrase_data[pr];
			const auto s_size = s.GetSize();
			if (s_size == 0) {
				continue;
			}

			if (!full) {
				l.topk.push_back({timestamp_t(ts_val), std::string(s.GetData(), s_size)});
				std::sort(l.topk.begin(), l.topk.end());
				if (l.topk.size() == 10) {
					full = true;
					max_ts = l.topk.back().ts.value;
				}
			} else {
				if (ts_val < max_ts || (ts_val == max_ts && LexCompareStringT(s, l.topk.back().phrase) < 0)) {
					l.topk.back().ts.value = ts_val;
					l.topk.back().phrase.assign(s.GetData(), s_size);
					for (int j = 8; j >= 0; --j) {
						if (l.topk[j + 1] < l.topk[j]) {
							std::swap(l.topk[j], l.topk[j + 1]);
						} else {
							break;
						}
					}
					max_ts = l.topk.back().ts.value;
				}
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize: merge local -> global; last thread emits sorted top-10
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			for (auto &le : l.topk) {
				if (g.topk.size() < 10) {
					g.topk.push_back(std::move(le));
					std::sort(g.topk.begin(), g.topk.end());
				} else if (le < g.topk.back()) {
					g.topk.back() = std::move(le);
					for (int j = 8; j >= 0; --j) {
						if (g.topk[j + 1] < g.topk[j]) {
							std::swap(g.topk[j], g.topk[j + 1]);
						} else {
							break;
						}
					}
				}
			}
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active > 0 && merged == active) {
		std::vector<Entry> results;
		{
			std::lock_guard<std::mutex> guard(g.lock);
			results = g.topk;
		}
		out.SetCardinality(results.size());
		for (idx_t i = 0; i < (idx_t)results.size(); i++) {
			out.SetValue(0, i, Value(results[i].phrase));
		}
	} else {
		out.SetCardinality(0);
	}

	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
//  Register
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
