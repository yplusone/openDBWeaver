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
#include <array>

namespace duckdb {

// ============================================================
//  Helpers: Fixed sorted array for 10 smallest strings
// ============================================================

struct TopKSmallest {
	std::string data[10];
	idx_t count = 0;

	void Insert(const char* ptr, uint32_t len) {
		if (len == 0) return;
		string_t s(ptr, len);
		if (count < 10) {
			idx_t i = count;
			while (i > 0 && s < string_t(data[i - 1].data(), (uint32_t)data[i - 1].size())) {
				data[i] = std::move(data[i - 1]);
				i--;
			}
			data[i].assign(ptr, len);
			count++;
		} else {
			// Gate check: only call Insert if s < data[9]
			if (s < string_t(data[9].data(), (uint32_t)data[9].size())) {
				// Reuse string buffer to avoid heap overhead
				std::string to_insert = std::move(data[9]);
				to_insert.assign(ptr, len);
				idx_t i = 9;
				while (i > 0 && string_t(to_insert.data(), (uint32_t)to_insert.size()) < string_t(data[i - 1].data(), (uint32_t)data[i - 1].size())) {
					data[i] = std::move(data[i - 1]);
					i--;
				}
				data[i] = std::move(to_insert);
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
//  Global / Local state
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;
	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};

	TopKSmallest topk;

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	TopKSmallest topk;
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

	// Cache current max as gate to avoid overhead inside row loops
	const char* max_ptr = nullptr;
	uint32_t max_len = 0;
	bool full = (l.topk.count == 10);
	if (full) {
		max_ptr = l.topk.data[9].data();
		max_len = (uint32_t)l.topk.data[9].size();
	}

	auto process_string = [&](string_t s) {
		const auto slen = s.GetSize();
		if (slen == 0) return;
		if (full) {
			if (s < string_t(max_ptr, max_len)) {
				l.topk.Insert(s.GetData(), slen);
				max_ptr = l.topk.data[9].data();
				max_len = (uint32_t)l.topk.data[9].size();
			}
		} else {
			l.topk.Insert(s.GetData(), slen);
			full = (l.topk.count == 10);
			if (full) {
				max_ptr = l.topk.data[9].data();
				max_len = (uint32_t)l.topk.data[9].size();
			}
		}
	};

	switch (vec.GetVectorType()) {
	case VectorType::FLAT_VECTOR: {
		auto data = FlatVector::GetData<string_t>(vec);
		auto &validity = FlatVector::Validity(vec);
		for (idx_t i = 0; i < count; i++) {
			if (validity.RowIsValid(i)) {
				process_string(data[i]);
			}
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
				if (dict_validity.RowIsValid(dict_idx)) {
					process_string(dict_data[dict_idx]);
				}
			}
		} else {
			UnifiedVectorFormat uf;
			vec.ToUnifiedFormat(count, uf);
			auto data = (const string_t *)uf.data;
			for (idx_t i = 0; i < count; i++) {
				const auto rid = uf.sel->get_index(i);
				if (uf.validity.RowIsValid(rid)) {
					process_string(data[rid]);
				}
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
			if (uf.validity.RowIsValid(rid)) {
				process_string(data[rid]);
			}
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

	bool is_last = false;
	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			for (idx_t i = 0; i < l.topk.count; i++) {
				g.topk.Insert(l.topk.data[i].data(), (uint32_t)l.topk.data[i].size());
			}
		}
		l.merged = true;
		// Safely identify the last thread to finish merging
		if (g.merged_local_states.fetch_add(1, std::memory_order_relaxed) == g.active_local_states.load(std::memory_order_relaxed) - 1) {
			is_last = true;
		}
	}

	if (is_last) {
		// Output sorted top-10 directly from global array (already kept sorted)
		out.SetCardinality(g.topk.count);
		for (idx_t i = 0; i < g.topk.count; i++) {
			out.SetValue(0, i, Value(g.topk.data[i]));
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
