/*
query_template: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));
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
//  Helpers: keep 10 lexicographically smallest strings
// ============================================================

// max-heap by lexicographic order (top = current largest among kept)
struct MaxLexCmp {
	bool operator()(const std::string &a, const std::string &b) const noexcept {
		return a < b;
	}
};

static inline bool StrLessThan(const string_t &s, const std::string &t) noexcept {
	// lexicographic compare without allocating
	const auto slen = (idx_t)s.GetSize();
	const auto tlen = (idx_t)t.size();
	const auto n = slen < tlen ? slen : tlen;
	int cmp = 0;
	if (n > 0) {
		cmp = std::memcmp(s.GetData(), t.data(), n);
	}
	if (cmp != 0) {
		return cmp < 0;
	}
	return slen < tlen;
}

static inline void InsertTopKSmallest(std::priority_queue<std::string, std::vector<std::string>, MaxLexCmp> &pq,
                                     const string_t &s) {
	if (s.GetSize() == 0) {
		return;
	}
	if (pq.size() < 10) {
		pq.emplace(s.GetData(), s.GetSize());
		return;
	}
	// keep only if s < current largest
	if (StrLessThan(s, pq.top())) {
		pq.pop();
		pq.emplace(s.GetData(), s.GetSize());
	}
}

static inline void InsertTopKSmallest(std::priority_queue<std::string, std::vector<std::string>, MaxLexCmp> &pq,
                                     std::string &&s) {
	if (s.empty()) {
		return;
	}
	if (pq.size() < 10) {
		pq.push(std::move(s));
		return;
	}
	if (s < pq.top()) {
		pq.pop();
		pq.push(std::move(s));
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

	std::priority_queue<std::string, std::vector<std::string>, MaxLexCmp> topk;

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	std::priority_queue<std::string, std::vector<std::string>, MaxLexCmp> topk;
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

	D_ASSERT(input.ColumnCount() == 1);
	auto &vec = input.data[0];
	const auto count = input.size();

	switch (vec.GetVectorType()) {
	case VectorType::FLAT_VECTOR: {
		auto data = FlatVector::GetData<string_t>(vec);
		auto &validity = FlatVector::Validity(vec);
		for (idx_t i = 0; i < count; i++) {
			if (!validity.RowIsValid(i)) {
				continue;
			}
			InsertTopKSmallest(l.topk, data[i]);
		}
		break;
	}
	case VectorType::DICTIONARY_VECTOR: {
		auto &child = DictionaryVector::Child(vec);
		auto &sel = DictionaryVector::SelVector(vec);

		// fast path when child is flat
		if (child.GetVectorType() == VectorType::FLAT_VECTOR) {
			auto dict_data = FlatVector::GetData<string_t>(child);
			auto &dict_validity = FlatVector::Validity(child);
			for (idx_t i = 0; i < count; i++) {
				const auto dict_idx = sel.get_index(i);
				if (!dict_validity.RowIsValid(dict_idx)) {
					continue;
				}
				InsertTopKSmallest(l.topk, dict_data[dict_idx]);
			}
		} else {
			// fallback: unify only if needed
			UnifiedVectorFormat uf;
			vec.ToUnifiedFormat(count, uf);
			auto data = (const string_t *)uf.data;
			for (idx_t i = 0; i < count; i++) {
				const auto rid = uf.sel->get_index(i);
				if (!uf.validity.RowIsValid(rid)) {
					continue;
				}
				InsertTopKSmallest(l.topk, data[rid]);
			}
		}
		break;
	}


	default: {
		// generic fallback
		UnifiedVectorFormat uf;
		vec.ToUnifiedFormat(count, uf);
		auto data = (const string_t *)uf.data;
		for (idx_t i = 0; i < count; i++) {
			const auto rid = uf.sel->get_index(i);
			if (!uf.validity.RowIsValid(rid)) {
				continue;
			}
			InsertTopKSmallest(l.topk, data[rid]);
		}
		break;
	}
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
				InsertTopKSmallest(g.topk, std::move(const_cast<std::string &>(l.topk.top())));
				l.topk.pop();
			}
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	// only last merged local emits
	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active > 0 && merged == active) {
		std::vector<std::string> results;
		{
			std::lock_guard<std::mutex> guard(g.lock);
			results.reserve(g.topk.size());
			while (!g.topk.empty()) {
				results.emplace_back(std::move(const_cast<std::string &>(g.topk.top())));
				g.topk.pop();
			}
		}
		std::sort(results.begin(), results.end()); // ascending
		if (results.size() > 10) {
			results.resize(10);
		}

		out.SetCardinality(results.size());
		for (idx_t i = 0; i < (idx_t)results.size(); i++) {
			out.SetValue(0, i, Value(results[i]));
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