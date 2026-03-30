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
#include <queue>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

namespace duckdb {

// ============================================================
//  Helpers: keep 10 smallest tuples by (EventTime ASC, SearchPhrase ASC)
// ============================================================

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
	// prefix equal -> shorter is smaller
	if (alen < blen)
		return -1;
	if (alen > blen)
		return 1;
	return 0;
}

static inline int LexCompareStringT(const string_t &a, const string_t &b) noexcept {
	const auto alen = (idx_t)a.GetSize();
	const auto blen = (idx_t)b.GetSize();
	const auto n = alen < blen ? alen : blen;
	int cmp = 0;
	if (n > 0) {
		cmp = std::memcmp(a.GetData(), b.GetData(), n);
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

// For priority_queue: make TOP be the current "largest" among kept (max-heap w.r.t. Entry::operator<)
struct EntryMaxCmp {
	bool operator()(const Entry &a, const Entry &b) const noexcept {
		return a < b; // same pattern as earlier string top-k code
	}
};

static inline bool TupleLessThan(const timestamp_t ts, const string_t &phrase, const Entry &top) noexcept {
	if (ts.value != top.ts.value) {
		return ts.value < top.ts.value;
	}
	return LexCompareStringT(phrase, top.phrase) < 0;
}

static inline void InsertTopKSmallest(std::priority_queue<Entry, std::vector<Entry>, EntryMaxCmp> &pq,
                                     const timestamp_t ts, const string_t &phrase) {
	if (phrase.GetSize() == 0) {
		return;
	}
	if (pq.size() < 10) {
		pq.push(Entry{ts, std::string(phrase.GetData(), phrase.GetSize())});
		return;
	}
	// keep only if (ts, phrase) is smaller than current largest kept
	if (TupleLessThan(ts, phrase, pq.top())) {
		pq.pop();
		pq.push(Entry{ts, std::string(phrase.GetData(), phrase.GetSize())});
	}
}

static inline void InsertTopKSmallest(std::priority_queue<Entry, std::vector<Entry>, EntryMaxCmp> &pq, Entry &&e) {
	if (e.phrase.empty()) {
		return;
	}
	if (pq.size() < 10) {
		pq.push(std::move(e));
		return;
	}
	// keep only if e < top
	if (e < pq.top()) {
		pq.pop();
		pq.push(std::move(e));
	}
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

	std::priority_queue<Entry, std::vector<Entry>, EntryMaxCmp> topk;

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	std::priority_queue<Entry, std::vector<Entry>, EntryMaxCmp> topk;
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
	// output only SearchPhrase (matches query_template)
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
	auto &phrase_vec = input.data[0]; // SearchPhrase
	auto &time_vec = input.data[1];   // EventTime
	const auto count = input.size();

	// Generic (fast enough): unify both columns once per chunk, then iterate rows
	UnifiedVectorFormat phrase_uf;
	UnifiedVectorFormat time_uf;
	phrase_vec.ToUnifiedFormat(count, phrase_uf);
	time_vec.ToUnifiedFormat(count, time_uf);

	auto phrase_data = (const string_t *)phrase_uf.data;
	auto time_data = (const timestamp_t *)time_uf.data;

	for (idx_t i = 0; i < count; i++) {
		const auto pr = phrase_uf.sel->get_index(i);
		const auto tr = time_uf.sel->get_index(i);

		if (!phrase_uf.validity.RowIsValid(pr) || !time_uf.validity.RowIsValid(tr)) {
			continue;
		}
		const auto &s = phrase_data[pr];
		if (s.GetSize() == 0) {
			continue; // WHERE SearchPhrase <> ''
		}
		InsertTopKSmallest(l.topk, time_data[tr], s);
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize: merge local -> global; last thread emits sorted top-10
// ============================================================

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
	auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
	auto &l = in.local_state->Cast<FnLocalState>();

	// merge exactly once per local
	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			while (!l.topk.empty()) {
				auto e = std::move(const_cast<Entry &>(l.topk.top()));
				l.topk.pop();
				InsertTopKSmallest(g.topk, std::move(e));
			}
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	// only last merged local emits
	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active > 0 && merged == active) {
		std::vector<Entry> results;
		{
			std::lock_guard<std::mutex> guard(g.lock);
			results.reserve(g.topk.size());
			while (!g.topk.empty()) {
				results.emplace_back(std::move(const_cast<Entry &>(g.topk.top())));
				g.topk.pop();
			}
		}
		// sort ascending by (EventTime, SearchPhrase)
		std::sort(results.begin(), results.end(),
		          [](const Entry &a, const Entry &b) noexcept { return a < b; });
		if (results.size() > 10) {
			results.resize(10);
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