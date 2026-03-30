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
//  Helpers: keep 10 smallest tuples by (EventTime ASC, SearchPhrase ASC)
// ============================================================

struct Entry {
	timestamp_t ts;
	uint32_t len;
	char buf[256];

	void SetPhrase(const char *data, idx_t length) {
		len = (uint32_t)(length > 256 ? 256 : length);
		if (len > 0) {
			std::memcpy(buf, data, len);
		}
	}

	// lex ascending order
	inline bool operator<(const Entry &other) const noexcept {
		if (ts.value != other.ts.value) {
			return ts.value < other.ts.value;
		}
		const auto n = len < other.len ? len : other.len;
		int cmp = 0;
		if (n > 0) {
			cmp = std::memcmp(buf, other.buf, (size_t)n);
		}
		if (cmp != 0) {
			return cmp < 0;
		}
		return len < other.len;
	}
};

static inline int FindMaxIdx(const Entry *arr, int sz) {
	if (sz == 0) return -1;
	int m = 0;
	for (int i = 1; i < sz; i++) {
		if (arr[m] < arr[i]) {
			m = i;
		}
	}
	return m;
}

static inline void InsertTopK(Entry *arr, int &sz, int &max_idx, Entry &&e) {
	if (sz < 10) {
		arr[sz++] = std::move(e);
		max_idx = FindMaxIdx(arr, sz);
	} else if (e < arr[max_idx]) {
		arr[max_idx] = std::move(e);
		max_idx = FindMaxIdx(arr, sz);
	}
}

static inline int LexCompareStringT(const string_t &a, const Entry &b) noexcept {
	const auto alen = (idx_t)a.GetSize();
	const auto blen = (idx_t)b.len;
	const auto n = alen < blen ? alen : blen;
	int cmp = 0;
	if (n > 0) {
		cmp = std::memcmp(a.GetData(), b.buf, n);
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

	Entry topk_arr[10];
	int topk_sz = 0;
	int max_idx = -1;

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	Entry topk_arr[10];
	int topk_sz = 0;
	int max_idx = -1;
	int64_t max_ts_threshold = std::numeric_limits<int64_t>::max();
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
	const auto count = input.size();
	if (count == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	D_ASSERT(input.ColumnCount() == 2);
	auto &phrase_vec = input.data[0]; // SearchPhrase
	auto &time_vec = input.data[1];   // EventTime
	if (phrase_vec.GetVectorType() == VectorType::FLAT_VECTOR && time_vec.GetVectorType() == VectorType::FLAT_VECTOR) {
		// Fast path: bypass UnifiedVectorFormat selection indirection for FLAT_VECTOR inputs
		auto phrase_data = FlatVector::GetData<string_t>(phrase_vec);
		auto time_data = FlatVector::GetData<timestamp_t>(time_vec);
		auto &phrase_validity = FlatVector::Validity(phrase_vec);
		auto &time_validity = FlatVector::Validity(time_vec);

		uint16_t survivors[1024];
		idx_t nsurv = 0;
		const int64_t threshold = l.max_ts_threshold;

		// Pass 1: Tight scalar loop touching only contiguous timestamp array
		for (idx_t i = 0; i < count; i++) {
			if (time_data[i].value <= threshold) {
				survivors[nsurv++] = (uint16_t)i;
			}
		}

		// Pass 2: Process only survivor rows (validity, string fetching, heap ops)
		for (idx_t j = 0; j < nsurv; j++) {
			const auto i = survivors[j];
			const auto ts_val = time_data[i].value;

			if (ts_val > l.max_ts_threshold) {
				continue;
			}

			if (!time_validity.RowIsValid(i) || !phrase_validity.RowIsValid(i)) {
				continue;
			}

			const auto &s = phrase_data[i];
			const auto s_len = s.GetSize();
			if (s_len == 0) {
				continue; 
			}

			if (l.topk_sz < 10) {
				Entry e;
				e.ts = time_data[i];
				e.SetPhrase(s.GetData(), s_len);
				l.topk_arr[l.topk_sz++] = std::move(e);
				l.max_idx = FindMaxIdx(l.topk_arr, l.topk_sz);
				if (l.topk_sz == 10) {
					l.max_ts_threshold = l.topk_arr[l.max_idx].ts.value;
				}
			} else {
				const auto &top = l.topk_arr[l.max_idx];
				if (ts_val < l.max_ts_threshold || LexCompareStringT(s, top) < 0) {
					Entry e;
					e.ts = time_data[i];
					e.SetPhrase(s.GetData(), s_len);
					l.topk_arr[l.max_idx] = std::move(e);
					l.max_idx = FindMaxIdx(l.topk_arr, l.topk_sz);
					l.max_ts_threshold = l.topk_arr[l.max_idx].ts.value;
				}
			}
		}
	} else {

		// Generic fallback: unify both columns once per chunk, then iterate rows
		UnifiedVectorFormat phrase_uf;
		UnifiedVectorFormat time_uf;
		phrase_vec.ToUnifiedFormat(count, phrase_uf);
		time_vec.ToUnifiedFormat(count, time_uf);

		auto phrase_data = (const string_t *)phrase_uf.data;
		auto time_data = (const timestamp_t *)time_uf.data;
		for (idx_t i = 0; i < count; i++) {
			const auto tr = time_uf.sel->get_index(i);
			const auto ts_val = time_data[tr].value;

			// O(1) Rejection check
			if (ts_val > l.max_ts_threshold) {
				continue;
			}

			const auto pr = phrase_uf.sel->get_index(i);
			if (!time_uf.validity.RowIsValid(tr) || !phrase_uf.validity.RowIsValid(pr)) {
				continue;
			}

			const auto &s = phrase_data[pr];
			const auto s_len = s.GetSize();
			if (s_len == 0) {
				continue;
			}

			if (l.topk_sz < 10) {
				Entry e;
				e.ts = time_data[tr];
				e.SetPhrase(s.GetData(), s_len);
				l.topk_arr[l.topk_sz++] = std::move(e);
				l.max_idx = FindMaxIdx(l.topk_arr, l.topk_sz);
				if (l.topk_sz == 10) {
					l.max_ts_threshold = l.topk_arr[l.max_idx].ts.value;
				}
			} else {
				const auto &top = l.topk_arr[l.max_idx];
				if (ts_val < l.max_ts_threshold || LexCompareStringT(s, top) < 0) {
					Entry e;
					e.ts = time_data[tr];
					e.SetPhrase(s.GetData(), s_len);
					l.topk_arr[l.max_idx] = std::move(e);
					l.max_idx = FindMaxIdx(l.topk_arr, l.topk_sz);
					l.max_ts_threshold = l.topk_arr[l.max_idx].ts.value;
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
	auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
	auto &l = in.local_state->Cast<FnLocalState>();

	// merge exactly once per local
	if (!l.merged) {
		{
			std::lock_guard<std::mutex> guard(g.lock);
			for (int i = 0; i < l.topk_sz; i++) {
				InsertTopK(g.topk_arr, g.topk_sz, g.max_idx, std::move(l.topk_arr[i]));
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
			results.reserve(g.topk_sz);
			for (int i = 0; i < g.topk_sz; i++) {
				results.emplace_back(std::move(g.topk_arr[i]));
			}
		}
		// sort ascending by (EventTime, SearchPhrase)
		std::sort(results.begin(), results.end(),
		          [](const Entry &a, const Entry &b) noexcept { return a < b; });
		if (results.size() > 10) {
			results.resize(10);
		}
		const idx_t res_count = results.size();
		out.SetCardinality(res_count);
		auto result_data = FlatVector::GetData<string_t>(out.data[0]);
		for (idx_t i = 0; i < res_count; i++) {
			result_data[i] = StringVector::AddStringOrBlob(out.data[0], results[i].buf, results[i].len);
		}
		FlatVector::Validity(out.data[0]).SetAllValid(res_count);

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
