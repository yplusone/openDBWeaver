/*
query_template: SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase, UserID FROM hits WHERE (SearchPhrase!='')))
query_example: SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase, UserID FROM hits WHERE (SearchPhrase!='')))
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/hash.hpp"

#include <absl/container/flat_hash_set.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <algorithm>
#include <memory>
#include <functional>

namespace duckdb {

using user_id_t = int64_t;

// Hybrid distinct set for UserIDs: empty -> single -> sorted vector -> flat_hash_set
struct UserSet {
	union {
		user_id_t single;
		std::vector<user_id_t>* vec;
		absl::flat_hash_set<user_id_t>* set;
	} data;
	uint32_t size = 0;
	uint8_t type = 0; // 0: empty, 1: single, 2: vector, 3: set

	UserSet() : size(0), type(0) {}

	void Destroy() {
		if (type == 2) delete data.vec;
		else if (type == 3) delete data.set;
		type = 0; size = 0;
	}
	void Insert(user_id_t uid) {
		if (type == 0) {
			data.single = uid; size = 1; type = 1;
		} else if (type == 1) {
			if (data.single != uid) {
				auto* v = new std::vector<user_id_t>();
				v->reserve(8);
				v->push_back(data.single); v->push_back(uid);
				data.vec = v; size = 2; type = 2;
			}
		} else if (type == 2) {
			auto& v = *data.vec;
			for (auto x : v) { if (x == uid) return; }
			if (v.size() < 32) {
				v.push_back(uid); size = (uint32_t)v.size();
			} else {
				auto* s = new absl::flat_hash_set<user_id_t>(v.begin(), v.end());
				s->insert(uid);
				delete data.vec;
				data.set = s; type = 3; size = (uint32_t)s->size();
			}
		} else {
			if (data.set->insert(uid).second) size = (uint32_t)data.set->size();
		}
	}


	void MergeInto(UserSet& other) {
		if (type == 0) return;
		if (other.type == 0) {
			other.type = type; other.size = size; other.data = data;
			type = 0; return;
		}
		if (type == 1) other.Insert(data.single);
		else if (type == 2) { for (auto uid : *data.vec) other.Insert(uid); }
		else { for (auto uid : *data.set) other.Insert(uid); }
	}
};

struct PhraseEntry {
	string_t str;
	uint64_t hash;
	UserSet users;
};

struct PhraseSlot {
	uint64_t hash;
	uint32_t phrase_id;
};

struct PhraseLookupTable {
	PhraseSlot* slots = nullptr;
	uint32_t capacity = 0, mask = 0, size = 0;

	PhraseLookupTable() = default;
	PhraseLookupTable(const PhraseLookupTable&) = delete;
	PhraseLookupTable(PhraseLookupTable&& o) noexcept : slots(o.slots), capacity(o.capacity), mask(o.mask), size(o.size) {
		o.slots = nullptr; o.capacity = 0;
	}
	PhraseLookupTable& operator=(PhraseLookupTable&& o) noexcept {
		if (slots) delete[] slots;
		slots = o.slots; capacity = o.capacity; mask = o.mask; size = o.size;
		o.slots = nullptr; o.capacity = 0; return *this;
	}
	~PhraseLookupTable() { if (slots) delete[] slots; }

	void Resize(uint32_t new_cap) {
		PhraseSlot* old_slots = slots; uint32_t old_cap = capacity;
		capacity = new_cap; mask = capacity - 1;
		slots = new PhraseSlot[capacity];
		for (uint32_t i = 0; i < capacity; ++i) slots[i].phrase_id = 0xFFFFFFFF;
		for (uint32_t i = 0; i < old_cap; ++i) {
			if (old_slots[i].phrase_id != 0xFFFFFFFF) InsertInternal(old_slots[i].hash, old_slots[i].phrase_id);
		}
		if (old_slots) delete[] old_slots;
	}

	void InsertInternal(uint64_t h, uint32_t pid) {
		uint32_t idx = (uint32_t)(h & mask), dist = 0;
		while (true) {
			if (slots[idx].phrase_id == 0xFFFFFFFF) { slots[idx] = {h, pid}; return; }
			uint32_t ideal = (uint32_t)(slots[idx].hash & mask), s_dist = (idx - ideal) & mask;
			if (dist > s_dist) { std::swap(h, slots[idx].hash); std::swap(pid, slots[idx].phrase_id); dist = s_dist; }
			idx = (idx + 1) & mask; dist++;
		}
	}

	uint32_t FindOrInsert(const string_t& s, uint64_t h, std::vector<PhraseEntry>& entries, StringHeap& heap) {
		if (capacity == 0) Resize(1024);
		uint32_t idx = (uint32_t)(h & mask), dist = 0;
		while (true) {
			if (slots[idx].phrase_id == 0xFFFFFFFF) break;
			uint32_t ideal = (uint32_t)(slots[idx].hash & mask), s_dist = (idx - ideal) & mask;
			if (dist > s_dist) break;
			if (slots[idx].hash == h) {
				uint32_t pid = slots[idx].phrase_id;
				const string_t& estr = entries[pid].str;
				if (s.GetSize() == estr.GetSize() && (s.GetSize() == 0 || memcmp(s.GetData(), estr.GetData(), s.GetSize()) == 0)) return pid;
			}
			idx = (idx + 1) & mask; dist++;
		}
		if (size * 10 >= capacity * 7) { Resize(capacity * 2); return FindOrInsert(s, h, entries, heap); }
		uint32_t new_pid = (uint32_t)entries.size();
		entries.push_back({heap.AddString(s), h, UserSet()});
		InsertInternal(h, new_pid); size++; return new_pid;
	}
};

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;
	unique_ptr<StringHeap> heap;
	PhraseLookupTable lookup;
	std::vector<PhraseEntry> entries;
	std::atomic<idx_t> active_local_states{0}, merged_local_states{0};
	std::atomic<uint8_t> adopt_stage{0};

	FnGlobalState() { heap = make_uniq<StringHeap>(); }
	~FnGlobalState() { for (auto& e : entries) e.users.Destroy(); }
	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	unique_ptr<StringHeap> heap;
	PhraseLookupTable lookup;
	std::vector<PhraseEntry> entries;
	bool merged = false;

	FnLocalState() { heap = make_uniq<StringHeap>(); }
	~FnLocalState() { if (!merged) { for (auto& e : entries) e.users.Destroy(); } }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) { return make_uniq<FnGlobalState>(); }
static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &, GlobalTableFunctionState *gs) {
	gs->Cast<FnGlobalState>().active_local_states.fetch_add(1, std::memory_order_relaxed);
	return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR); return_types.push_back(LogicalType::BIGINT);
	names.push_back("SearchPhrase"); names.push_back("u"); return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;
	auto &l = in.local_state->Cast<FnLocalState>();
	UnifiedVectorFormat uvf0, uvf1;
	input.data[0].ToUnifiedFormat(input.size(), uvf0); input.data[1].ToUnifiedFormat(input.size(), uvf1);
	auto *ptr0 = (string_t *)uvf0.data; auto *ptr1 = (user_id_t *)uvf1.data;
	constexpr idx_t BATCH = 8;
	for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
		idx_t n = std::min(BATCH, input.size() - rr);
		string_t batch_strs[BATCH]; uint64_t batch_hashes[BATCH]; user_id_t batch_uids[BATCH];
		for (idx_t k = 0; k < n; ++k) {
			idx_t i0 = uvf0.sel->get_index(rr + k), i1 = uvf1.sel->get_index(rr + k);
			if (!uvf0.validity.RowIsValid(i0) || !uvf1.validity.RowIsValid(i1)) { batch_hashes[k] = 0; continue; }
			batch_strs[k] = ptr0[i0]; batch_uids[k] = ptr1[i1];
			if (batch_strs[k].GetSize() == 0) batch_hashes[k] = 0;
			else batch_hashes[k] = duckdb::Hash(batch_strs[k].GetData(), batch_strs[k].GetSize());
		}
		for (idx_t k = 0; k < n; ++k) {
			if (batch_hashes[k] == 0) continue;
#ifdef __GNUC__
			__builtin_prefetch(&l.lookup.slots[batch_hashes[k] & l.lookup.mask], 0, 1);
#endif
		}
		for (idx_t k = 0; k < n; ++k) {
			if (batch_hashes[k] == 0) continue;
			uint32_t pid = l.lookup.FindOrInsert(batch_strs[k], batch_hashes[k], l.entries, *l.heap);
			l.entries[pid].users.Insert(batch_uids[k]);
		}
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

static inline bool LexLessThan(const string_t &a, const string_t &b) {
	const auto an = a.GetSize(); const auto bn = b.GetSize();
	const auto mn = std::min(an, bn);
	int cmp = mn ? std::memcmp(a.GetData(), b.GetData(), mn) : 0;
	if (cmp != 0) return cmp < 0;
	return an < bn;
}

struct TopRow {
	uint32_t u; string_t str;
};

struct TopRowCmp {
	// Priority queue top is the "worst" of the top 10. 
	// ORDER BY u DESC, str ASC -> Better: (higher u) or (same u and lower str).
	// Worse: (lower u) or (same u and higher str).
	bool operator()(const TopRow& a, const TopRow& b) const {
		if (a.u != b.u) return a.u > b.u; // Higher u is better -> a is better -> b is worse (closer to top of min-heap)
		return LexLessThan(a.str, b.str); // Lower str is better -> a is better -> b is worse
	}
};

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>(); auto &l = in.local_state->Cast<FnLocalState>();
	const auto active = g.active_local_states.load(std::memory_order_relaxed);
	if (active == 1) {
		std::priority_queue<TopRow, std::vector<TopRow>, TopRowCmp> pq;
		for (auto &e : l.entries) {
			TopRow tr{e.users.size, e.str};
			if (pq.size() < 10) pq.push(tr); else if (tr.u > pq.top().u || (tr.u == pq.top().u && LexLessThan(tr.str, pq.top().str))) { pq.pop(); pq.push(tr); }
		}
		std::vector<TopRow> top; while (!pq.empty()) { top.push_back(pq.top()); pq.pop(); }
		std::sort(top.begin(), top.end(), [](const TopRow& a, const TopRow& b) {
			if (a.u != b.u) return a.u > b.u;
			return LexLessThan(a.str, b.str);
		});
		for (idx_t i = 0; i < top.size(); ++i) {
			FlatVector::GetData<string_t>(out.data[0])[i] = StringVector::AddString(out.data[0], top[i].str);
			FlatVector::GetData<int64_t>(out.data[1])[i] = (int64_t)top[i].u;
		}
		out.SetCardinality(top.size()); return OperatorFinalizeResultType::FINISHED;
	}
	if (!l.merged) {
		uint8_t expected = 0;
		if (g.adopt_stage.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> lock(g.lock);
			if (g.entries.empty()) { g.heap = std::move(l.heap); g.lookup = std::move(l.lookup); g.entries = std::move(l.entries); if (!g.heap) g.heap = make_uniq<StringHeap>(); }
			else { for (auto &le : l.entries) { uint32_t pid = g.lookup.FindOrInsert(le.str, le.hash, g.entries, *g.heap); le.users.MergeInto(g.entries[pid].users); le.users.Destroy(); } }
			g.adopt_stage.store(2, std::memory_order_release); l.merged = true; g.merged_local_states.fetch_add(1);
		} else {
			while (g.adopt_stage.load(std::memory_order_acquire) == 1) std::this_thread::yield();
			std::lock_guard<std::mutex> lock(g.lock);
			for (auto &le : l.entries) {
				uint32_t pid = g.lookup.FindOrInsert(le.str, le.hash, g.entries, *g.heap);
				le.users.MergeInto(g.entries[pid].users); le.users.Destroy();
			}
			l.merged = true; g.merged_local_states.fetch_add(1);
		}
	}
	if (g.merged_local_states.load(std::memory_order_relaxed) < active) { out.SetCardinality(0); return OperatorFinalizeResultType::FINISHED; }
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowCmp> pq;
	for (auto &e : g.entries) {
		TopRow tr{e.users.size, e.str};
		if (pq.size() < 10) pq.push(tr); else if (tr.u > pq.top().u || (tr.u == pq.top().u && LexLessThan(tr.str, pq.top().str))) { pq.pop(); pq.push(tr); }
	}
	std::vector<TopRow> top; while (!pq.empty()) { top.push_back(pq.top()); pq.pop(); }
	std::sort(top.begin(), top.end(), [](const TopRow& a, const TopRow& b) {
		if (a.u != b.u) return a.u > b.u;
		return LexLessThan(a.str, b.str);
	});
	for (idx_t i = 0; i < top.size(); ++i) {
		FlatVector::GetData<string_t>(out.data[0])[i] = StringVector::AddString(out.data[0], top[i].str);
		FlatVector::GetData<int64_t>(out.data[1])[i] = (int64_t)top[i].u;
	}
	out.SetCardinality(top.size()); return OperatorFinalizeResultType::FINISHED;
}

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
	f.in_out_function = FnExecute; f.in_out_function_final = FnFinalize; loader.RegisterFunction(f);
}

void DbweaverExtension::Load(ExtensionLoader &loader) { LoadInternal(loader); }
std::string DbweaverExtension::Name() { return "dbweaver"; }
std::string DbweaverExtension::Version() const { return DuckDB::LibraryVersion(); }

} // namespace duckdb

extern "C" { DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) { duckdb::LoadInternal(loader); } }