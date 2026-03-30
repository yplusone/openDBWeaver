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
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

namespace duckdb {

// ============================================================
//  Helpers: keep 10 lexicographically smallest strings
// ============================================================

static inline uint32_t GetPrefixUint32(const string_t &s) {
	uint32_t p = 0;
	std::memcpy(&p, s.GetPrefix(), 4);
#if (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || (defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN)
	return p;
#else
	return __builtin_bswap32(p);
#endif
}

struct StringEntry {
	std::string str;
	uint32_t prefix;

	StringEntry(std::string &&s_val, uint32_t p) : str(std::move(s_val)), prefix(p) {
	}

	bool operator<(const StringEntry &other) const {
		if (prefix != other.prefix) {
			return prefix < other.prefix;
		}
		return str < other.str;
	}
};

static inline bool StrLessThan(const string_t &s, uint32_t s_p, const StringEntry &t) noexcept {
	// Use the 4-byte prefix to short-circuit comparison
	if (s_p != t.prefix) {
		return s_p < t.prefix;
	}

	// Prefixes are identical, perform full comparison
	const auto slen = (idx_t)s.GetSize();
	const auto tlen = (idx_t)t.str.size();
	const auto n = slen < tlen ? slen : tlen;

	// We already know first 4 bytes match if n >= 4
	if (n > 4) {
		int cmp = std::memcmp(s.GetData() + 4, t.str.data() + 4, n - 4);
		if (cmp != 0) {
			return cmp < 0;
		}
	}
	return slen < tlen;
}

static inline void InsertTopKSmallest(std::vector<StringEntry> &vec, const string_t &s, uint32_t sp) {
	if (vec.size() < 10) {
		vec.emplace_back(std::string(s.GetData(), s.GetSize()), sp);
		std::sort(vec.begin(), vec.end());
		return;
	}
	// keep only if s < current largest (back of the sorted vector)
	if (StrLessThan(s, sp, vec.back())) {
		vec.pop_back();
		StringEntry entry(std::string(s.GetData(), s.GetSize()), sp);
		auto it = std::lower_bound(vec.begin(), vec.end(), entry);
		vec.insert(it, std::move(entry));
	}
}

static inline void InsertTopKSmallest(std::vector<StringEntry> &vec, StringEntry &&entry) {
	if (vec.size() < 10) {
		vec.push_back(std::move(entry));
		std::sort(vec.begin(), vec.end());
		return;
	}
	if (entry < vec.back()) {
		vec.pop_back();
		auto it = std::lower_bound(vec.begin(), vec.end(), entry);
		vec.insert(it, std::move(entry));
	}
}

static inline void ProcessString(std::vector<StringEntry> &topk, std::atomic<uint32_t> &global_threshold_p, const string_t &s) {
	const auto slen = s.GetSize();
	if (slen == 0) {
		return;
	}
	uint32_t sp = GetPrefixUint32(s);

	// Global Threshold Pruning: if the prefix is already larger than the best-found 10th-best prefix, skip immediately.
	if (sp > global_threshold_p.load(std::memory_order_relaxed)) {
		return;
	}

	InsertTopKSmallest(topk, s, sp);

	if (topk.size() == 10) {
		uint32_t local_p = topk.back().prefix;
		uint32_t current_global = global_threshold_p.load(std::memory_order_relaxed);
		// Update the global threshold if our local 10th-best prefix is smaller.
		while (local_p < current_global) {
			if (global_threshold_p.compare_exchange_weak(current_global, local_p, std::memory_order_relaxed)) {
				break;
			}
			// On failure, current_global is updated to the latest value of global_threshold_p
		}
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
	std::atomic<uint32_t> global_threshold_p {0xFFFFFFFF};

	std::vector<StringEntry> topk;

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	std::vector<StringEntry> topk;
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
	auto &g = in.global_state->Cast<FnGlobalState>();
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
			ProcessString(l.topk, g.global_threshold_p, data[i]);
		}
		break;
	}
	case VectorType::DICTIONARY_VECTOR: {
		auto &child = DictionaryVector::Child(vec);
		auto &sel = DictionaryVector::SelVector(vec);

		if (child.GetVectorType() == VectorType::FLAT_VECTOR) {
			auto dict_data = FlatVector::GetData<string_t>(child);
			auto &dict_validity = FlatVector::Validity(child);
			for (idx_t i = 0; i < count; i++) {
				const auto dict_idx = sel.get_index(i);
				if (!dict_validity.RowIsValid(dict_idx)) {
					continue;
				}
				ProcessString(l.topk, g.global_threshold_p, dict_data[dict_idx]);
			}
		} else {
			UnifiedVectorFormat uf;
			vec.ToUnifiedFormat(count, uf);
			auto data = (const string_t *)uf.data;
			for (idx_t i = 0; i < count; i++) {
				const auto rid = uf.sel->get_index(i);
				if (!uf.validity.RowIsValid(rid)) {
					continue;
				}
				ProcessString(l.topk, g.global_threshold_p, data[rid]);
			}
		}
		break;
	}

	default: {
		UnifiedVectorFormat uf;
		vec.ToUnifiedFormat(count, uf);
		auto data = (const string_t *)uf.data;
		for (idx_t i = 0; i < count; i++) {
			const auto rid = uf.sel->get_index(i);
			if (!uf.validity.RowIsValid(rid)) {
				continue;
			}
			ProcessString(l.topk, g.global_threshold_p, data[rid]);
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
			for (auto &entry : l.topk) {
				InsertTopKSmallest(g.topk, std::move(entry));
			}
			l.topk.clear();
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	// only last merged local emits
	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active > 0 && merged == active) {
		std::lock_guard<std::mutex> guard(g.lock);
		out.SetCardinality(g.topk.size());
		for (idx_t i = 0; i < (idx_t)g.topk.size(); i++) {
			out.SetValue(0, i, Value(g.topk[i].str));
		}
		// Clear to avoid re-emitting if Finalize is called multiple times
		g.topk.clear();
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
