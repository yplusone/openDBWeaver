/*
query_template: SELECT SearchPhrase, MIN(URL) AS min_url, COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT URL, SearchPhrase FROM hits WHERE (contains(URL, 'google')) AND (SearchPhrase!='')));
query_example: SELECT SearchPhrase, MIN(URL) AS min_url, COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT URL, SearchPhrase FROM hits WHERE (contains(URL, 'google')) AND (SearchPhrase!='')));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <algorithm>

namespace duckdb {

// ============================================================
//  Key wrapper: string_t + precomputed hash
// ============================================================

struct HashedStringT {
	string_t str;
	size_t hash;

	bool operator==(const HashedStringT &o) const noexcept {
		if (hash != o.hash) return false;
		const auto an = str.GetSize();
		const auto bn = o.str.GetSize();
		if (an != bn) return false;
		if (str.GetData() == o.str.GetData()) return true;
		return !an || std::memcmp(str.GetData(), o.str.GetData(), an) == 0;
	}
};

struct HashedStringTHash {
	size_t operator()(const HashedStringT &k) const noexcept { return k.hash; }
};

static inline size_t HashString(const char *p, idx_t n) { return duckdb::Hash(p, n); }

// ============================================================
//  Filters moved into extension
// ============================================================

static inline bool IsEmptyString(const string_t &s) noexcept { return s.GetSize() == 0; }

static inline bool ContainsLiteral(const string_t &hay, const char *needle, idx_t needle_len) noexcept {
	const idx_t n = hay.GetSize();
	if (needle_len == 0) return true;
	if (n < needle_len) return false;
	const char *p = hay.GetData();
	for (idx_t i = 0; i + needle_len <= n; i++) {
		if (p[i] == needle[0] && std::memcmp(p + i, needle, needle_len) == 0) return true;
	}
	return false;
}

static inline bool UrlPassesFilter(const string_t &url) noexcept {
	// URL LIKE '%google%'  <=> contains(url, "google")  (case-sensitive)
	static constexpr char NEEDLE[] = "google";
	return ContainsLiteral(url, NEEDLE, (idx_t)6);
}

static inline bool IsLexMin(const string_t &a, const string_t &b) {
	const auto an = a.GetSize();
	const auto bn = b.GetSize();
	const auto mn = std::min(an, bn);
	int cmp = mn ? std::memcmp(a.GetData(), b.GetData(), mn) : 0;
	if (cmp != 0) return cmp < 0;
	return an < bn;
}

// ============================================================
//  Aggregation state
// ============================================================

struct GroupAgg {
	string_t min_url;
	bool min_url_set;
	int64_t count;

	GroupAgg() : min_url_set(false), count(0) {}
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

	unique_ptr<StringHeap> heap;
	absl::flat_hash_map<HashedStringT, GroupAgg, HashedStringTHash> map;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};
	std::atomic<uint8_t> adopt_stage{0};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	unique_ptr<StringHeap> heap;
	absl::flat_hash_map<HashedStringT, GroupAgg, HashedStringTHash> map;
};

// ============================================================
//  Init / Bind
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	auto gs = make_uniq<FnGlobalState>();
	gs->heap = make_uniq<StringHeap>();
	return unique_ptr<GlobalTableFunctionState>(std::move(gs));
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);

	auto ls = make_uniq<FnLocalState>();
	ls->heap = make_uniq<StringHeap>();
	return unique_ptr<LocalTableFunctionState>(std::move(ls));
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types, vector<string> &names) {
	names.push_back("SearchPhrase");
	names.push_back("min_url");
	names.push_back("c");

	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::BIGINT);

	return make_uniq<FnBindData>();
}

// ============================================================
//  Execute
//  Input: URL, SearchPhrase
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	auto &l = in.local_state->Cast<FnLocalState>();
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	UnifiedVectorFormat url_uvf;
	UnifiedVectorFormat phrase_uvf;
	input.data[0].ToUnifiedFormat(input.size(), url_uvf);
	input.data[1].ToUnifiedFormat(input.size(), phrase_uvf);

	auto *url_ptr = (string_t *)url_uvf.data;
	auto *phrase_ptr = (string_t *)phrase_uvf.data;

	auto &valid_url = url_uvf.validity;
	auto &valid_phrase = phrase_uvf.validity;

	const bool url_all_valid = valid_url.AllValid();
	const bool phrase_all_valid = valid_phrase.AllValid();

	const idx_t n = input.size();

	if (url_all_valid && phrase_all_valid) {
		for (idx_t r = 0; r < n; r++) {
			const idx_t i0 = url_uvf.sel->get_index(r);
			const idx_t i1 = phrase_uvf.sel->get_index(r);

			const string_t url = url_ptr[i0];
			const string_t phrase = phrase_ptr[i1];

			if (!UrlPassesFilter(url)) continue;
			if (IsEmptyString(phrase)) continue;

			const size_t h = HashString(phrase.GetData(), phrase.GetSize());
			HashedStringT key{phrase, h};

			auto it = l.map.find(key);
			GroupAgg *agg_ptr = nullptr;

			if (it != l.map.end()) {
				agg_ptr = &it->second;
			} else {
				const auto owned_phrase = l.heap->AddString(phrase);
				agg_ptr = &l.map.emplace(HashedStringT{owned_phrase, h}, GroupAgg()).first->second;
			}

			if (!agg_ptr->min_url_set || IsLexMin(url, agg_ptr->min_url)) {
				agg_ptr->min_url = l.heap->AddString(url);
				agg_ptr->min_url_set = true;
			}
			agg_ptr->count += 1;
		}
	} else {
		for (idx_t r = 0; r < n; r++) {
			const idx_t i0 = url_uvf.sel->get_index(r);
			const idx_t i1 = phrase_uvf.sel->get_index(r);

			if (!url_all_valid && !valid_url.RowIsValid(i0)) continue;
			if (!phrase_all_valid && !valid_phrase.RowIsValid(i1)) continue;

			const string_t url = url_ptr[i0];
			const string_t phrase = phrase_ptr[i1];

			if (!UrlPassesFilter(url)) continue;
			if (IsEmptyString(phrase)) continue;

			const size_t h = HashString(phrase.GetData(), phrase.GetSize());
			HashedStringT key{phrase, h};

			auto it = l.map.find(key);
			GroupAgg *agg_ptr = nullptr;

			if (it != l.map.end()) {
				agg_ptr = &it->second;
			} else {
				const auto owned_phrase = l.heap->AddString(phrase);
				agg_ptr = &l.map.emplace(HashedStringT{owned_phrase, h}, GroupAgg()).first->second;
			}

			if (!agg_ptr->min_url_set || IsLexMin(url, agg_ptr->min_url)) {
				agg_ptr->min_url = l.heap->AddString(url);
				agg_ptr->min_url_set = true;
			}
			agg_ptr->count += 1;
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Merge helper
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (auto &kv : local.map) {
		const HashedStringT &k_hs = kv.first;
		const GroupAgg &local_agg = kv.second;
		if (local_agg.count == 0) continue;

		auto it = g.map.find(k_hs);
		if (it != g.map.end()) {
			GroupAgg &global_agg = it->second;

			if (!global_agg.min_url_set ||
			    (local_agg.min_url_set && IsLexMin(local_agg.min_url, global_agg.min_url))) {
				global_agg.min_url = g.heap->AddString(local_agg.min_url);
				global_agg.min_url_set = true;
			}
			global_agg.count += local_agg.count;
		} else {
			const auto owned_phrase = g.heap->AddString(k_hs.str);

			GroupAgg nagg;
			if (local_agg.min_url_set) {
				nagg.min_url = g.heap->AddString(local_agg.min_url);
				nagg.min_url_set = true;
			}
			nagg.count = local_agg.count;

			g.map.emplace(HashedStringT{owned_phrase, k_hs.hash}, std::move(nagg));
		}
	}
}

// ============================================================
//  Finalize: adopt-first + merge rest + Top10 by count
// ============================================================

struct TopRow {
	int64_t count;
	const string_t *phrase;
	const string_t *min_url;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const noexcept {
		// min-heap by count (priority_queue keeps "largest" by comparator false; so invert)
		return a.count > b.count;
	}
};

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	auto less_lex = [](const string_t &a, const string_t &b) {
		const auto an = a.GetSize();
		const auto bn = b.GetSize();
		const auto mn = std::min(an, bn);
		int cmp = mn ? std::memcmp(a.GetData(), b.GetData(), mn) : 0;
		if (cmp != 0) return cmp < 0;
		return an < bn;
	};

	// single-thread: directly output from local
	if (active == 1) {
		std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

		for (auto &kv : l.map) {
			const GroupAgg &agg = kv.second;
			if (agg.count <= 0) continue;

			if (pq.size() < 10) pq.push({agg.count, &kv.first.str, &agg.min_url});
			else if (agg.count > pq.top().count) {
				pq.pop();
				pq.push({agg.count, &kv.first.str, &agg.min_url});
			}
		}

		std::vector<TopRow> top;
		top.reserve(pq.size());
		while (!pq.empty()) {
			top.push_back(pq.top());
			pq.pop();
		}

		std::sort(top.begin(), top.end(), [&](const TopRow &a, const TopRow &b) {
			if (a.count != b.count) return a.count > b.count;
			return less_lex(*a.phrase, *b.phrase);
		});

		auto *out_phrase = FlatVector::GetData<string_t>(out.data[0]);
		auto *out_url = FlatVector::GetData<string_t>(out.data[1]);
		auto *out_count = FlatVector::GetData<int64_t>(out.data[2]);

		idx_t out_idx = 0;
		for (auto &r : top) {
			out_phrase[out_idx] = StringVector::AddString(out.data[0], *r.phrase);
			out_url[out_idx] = StringVector::AddString(out.data[1], *r.min_url);
			out_count[out_idx] = r.count;
			out_idx++;
		}
		out.SetCardinality(out_idx);
		return OperatorFinalizeResultType::FINISHED;
	}

	// multi-thread adopt/merge
	if (!l.merged) {
		uint8_t expected = 0;
		if (g.adopt_stage.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> guard(g.lock);
			if (g.map.empty()) {
				g.heap = std::move(l.heap);
				g.map = std::move(l.map);
				if (!g.heap) g.heap = make_uniq<StringHeap>();
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
	if (!(active > 0 && merged == active)) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	// output from global
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (auto &kv : g.map) {
		const GroupAgg &agg = kv.second;
		if (agg.count <= 0) continue;

		if (pq.size() < 10) pq.push({agg.count, &kv.first.str, &agg.min_url});
		else if (agg.count > pq.top().count) {
			pq.pop();
			pq.push({agg.count, &kv.first.str, &agg.min_url});
		}
	}

	std::vector<TopRow> top;
	top.reserve(pq.size());
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}

	std::sort(top.begin(), top.end(), [&](const TopRow &a, const TopRow &b) {
		if (a.count != b.count) return a.count > b.count;
		return less_lex(*a.phrase, *b.phrase);
	});

	auto *out_phrase = FlatVector::GetData<string_t>(out.data[0]);
	auto *out_url = FlatVector::GetData<string_t>(out.data[1]);
	auto *out_count = FlatVector::GetData<int64_t>(out.data[2]);

	idx_t out_idx = 0;
	for (auto &r : top) {
		out_phrase[out_idx] = StringVector::AddString(out.data[0], *r.phrase);
		out_url[out_idx] = StringVector::AddString(out.data[1], *r.min_url);
		out_count[out_idx] = r.count;
		out_idx++;
	}
	out.SetCardinality(out_idx);

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
DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) { duckdb::LoadInternal(loader); }
}