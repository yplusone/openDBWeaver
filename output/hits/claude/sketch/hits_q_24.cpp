/*
query_template: SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;

split_template: select * from dbweaver((SELECT URL, WatchID, JavaEnable, Title, GoodEvent, EventTime, EventDate, CounterID, ClientIP, RegionID, UserID, CounterClass, OS, UserAgent, Referer, IsRefresh, RefererCategoryID, RefererRegionID, URLCategoryID, URLRegionID, ResolutionWidth, ResolutionHeight, ResolutionDepth, FlashMajor, FlashMinor, FlashMinor2, NetMajor, NetMinor, UserAgentMajor, UserAgentMinor, CookieEnable, JavascriptEnable, IsMobile, MobilePhone, MobilePhoneModel, Params, IPNetworkID, TraficSourceID, SearchEngineID, SearchPhrase, AdvEngineID, IsArtifical, WindowClientWidth, WindowClientHeight, ClientTimeZone, ClientEventTime, SilverlightVersion1, SilverlightVersion2, SilverlightVersion3, SilverlightVersion4, PageCharset, CodeVersion, IsLink, IsDownload, IsNotBounce, FUniqID, OriginalURL, HID, IsOldCounter, IsEvent, IsParameter, DontCountHits, WithHash, HitColor, LocalEventTime, Age, Sex, Income, Interests, Robotness, RemoteIP, WindowName, OpenerName, HistoryLength, BrowserLanguage, BrowserCountry, SocialNetwork, SocialAction, HTTPError, SendTiming, DNSTiming, ConnectTiming, ResponseStartTiming, ResponseEndTiming, FetchTiming, SocialSourceNetworkID, SocialSourcePage, ParamPrice, ParamOrderID, ParamCurrency, ParamCurrencyID, OpenstatServiceName, OpenstatCampaignID, OpenstatAdID, OpenstatSourceID, UTMSource, UTMMedium, UTMCampaign, UTMContent, UTMTerm, FromTag, HasGCLID, RefererHash, URLHash, CLID FROM hits));
query_example: SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;

split_query: select * from dbweaver((SELECT URL, WatchID, JavaEnable, Title, GoodEvent, EventTime, EventDate, CounterID, ClientIP, RegionID, UserID, CounterClass, OS, UserAgent, Referer, IsRefresh, RefererCategoryID, RefererRegionID, URLCategoryID, URLRegionID, ResolutionWidth, ResolutionHeight, ResolutionDepth, FlashMajor, FlashMinor, FlashMinor2, NetMajor, NetMinor, UserAgentMajor, UserAgentMinor, CookieEnable, JavascriptEnable, IsMobile, MobilePhone, MobilePhoneModel, Params, IPNetworkID, TraficSourceID, SearchEngineID, SearchPhrase, AdvEngineID, IsArtifical, WindowClientWidth, WindowClientHeight, ClientTimeZone, ClientEventTime, SilverlightVersion1, SilverlightVersion2, SilverlightVersion3, SilverlightVersion4, PageCharset, CodeVersion, IsLink, IsDownload, IsNotBounce, FUniqID, OriginalURL, HID, IsOldCounter, IsEvent, IsParameter, DontCountHits, WithHash, HitColor, LocalEventTime, Age, Sex, Income, Interests, Robotness, RemoteIP, WindowName, OpenerName, HistoryLength, BrowserLanguage, BrowserCountry, SocialNetwork, SocialAction, HTTPError, SendTiming, DNSTiming, ConnectTiming, ResponseStartTiming, ResponseEndTiming, FetchTiming, SocialSourceNetworkID, SocialSourcePage, ParamPrice, ParamOrderID, ParamCurrency, ParamCurrencyID, OpenstatServiceName, OpenstatCampaignID, OpenstatAdID, OpenstatSourceID, UTMSource, UTMMedium, UTMCampaign, UTMContent, UTMTerm, FromTag, HasGCLID, RefererHash, URLHash, CLID FROM hits));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstring>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct SortKeyView {
    int64_t event_time;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        return a.event_time > b.event_time; // For min-heap (top-k keeps smallest)
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    std::vector<SelectionVector> row_indices;
    std::vector<DataChunk> chunks;
    bool sorted = false;
    
    inline void AddRow(int64_t event_time_val, idx_t chunk_idx, idx_t row_idx) {
        buffer.push_back(SortKeyView{event_time_val});
        
        // Store the reference to the original data
        if (row_indices.size() <= chunk_idx) {
            row_indices.resize(chunk_idx + 1);
        }
        row_indices[chunk_idx].set_index(buffer.size() - 1, row_idx);
    }
    
    inline void SortNow() {
        if (!sorted && !buffer.empty()) {
            std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    SortState sort_state;
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    std::vector<DataChunk> temp_chunks;
    //TODO: initialize local state and other preparations
    bool merged = false;
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                            vector<LogicalType> &return_types,
                                            vector<string> &names) {
    //TODO: populate return_types and names
    return_types.push_back(LogicalType::VARCHAR);   // URL
    return_types.push_back(LogicalType::BIGINT);   // WatchID
    return_types.push_back(LogicalType::SMALLINT); // JavaEnable
    return_types.push_back(LogicalType::VARCHAR);  // Title
    return_types.push_back(LogicalType::SMALLINT); // GoodEvent
    return_types.push_back(LogicalType::TIMESTAMP); // EventTime
    return_types.push_back(LogicalType::DATE);     // EventDate
    return_types.push_back(LogicalType::INTEGER);  // CounterID
    return_types.push_back(LogicalType::INTEGER);  // ClientIP
    return_types.push_back(LogicalType::INTEGER);  // RegionID
    return_types.push_back(LogicalType::BIGINT);   // UserID
    return_types.push_back(LogicalType::SMALLINT); // CounterClass
    return_types.push_back(LogicalType::SMALLINT); // OS
    return_types.push_back(LogicalType::SMALLINT); // UserAgent
    return_types.push_back(LogicalType::SMALLINT); // ParamCurrencyID
    return_types.push_back(LogicalType::VARCHAR);  // OpenstatServiceName
    return_types.push_back(LogicalType::VARCHAR);  // OpenstatCampaignID
    return_types.push_back(LogicalType::VARCHAR);  // OpenstatAdID
    return_types.push_back(LogicalType::VARCHAR);  // OpenstatSourceID
    return_types.push_back(LogicalType::VARCHAR);  // UTMSource
    return_types.push_back(LogicalType::VARCHAR);  // UTMMedium
    return_types.push_back(LogicalType::VARCHAR);  // UTMCampaign
    return_types.push_back(LogicalType::VARCHAR);  // UTMContent
    return_types.push_back(LogicalType::VARCHAR);  // UTMTerm
    return_types.push_back(LogicalType::VARCHAR);  // FromTag
    return_types.push_back(LogicalType::SMALLINT); // HasGCLID
    return_types.push_back(LogicalType::BIGINT);   // RefererHash
    return_types.push_back(LogicalType::BIGINT);   // URLHash
    return_types.push_back(LogicalType::INTEGER);  // CLID

    names.push_back("URL");
    names.push_back("WatchID");
    names.push_back("JavaEnable");
    names.push_back("Title");
    names.push_back("GoodEvent");
    names.push_back("EventTime");
    names.push_back("EventDate");
    names.push_back("CounterID");
    names.push_back("ClientIP");
    names.push_back("RegionID");
    names.push_back("UserID");
    names.push_back("CounterClass");
    names.push_back("OS");
    names.push_back("UserAgent");
    names.push_back("ParamCurrencyID");
    names.push_back("OpenstatServiceName");
    names.push_back("OpenstatCampaignID");
    names.push_back("OpenstatAdID");
    names.push_back("OpenstatSourceID");
    names.push_back("UTMSource");
    names.push_back("UTMMedium");
    names.push_back("UTMCampaign");
    names.push_back("UTMContent");
    names.push_back("UTMTerm");
    names.push_back("FromTag");
    names.push_back("HasGCLID");
    names.push_back("RefererHash");
    names.push_back("URLHash");
    names.push_back("CLID");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                DataChunk &input, DataChunk &output) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    //TODO: process input chunk and produce output
    
    // Get the URL column (index 0) for filtering
    auto &url_col = input.data[0];
    UnifiedVectorFormat url_uvf;
    url_col.ToUnifiedFormat(input.size(), url_uvf);
    auto url_ptr = UnifiedVectorFormat::GetData<string_t>(url_uvf);

    // Get the EventTime column (index 5) for sorting
    auto &event_time_col = input.data[5];
    UnifiedVectorFormat event_time_uvf;
    event_time_col.ToUnifiedFormat(input.size(), event_time_uvf);
    auto event_time_ptr = UnifiedVectorFormat::GetData<timestamp_t>(event_time_uvf);

    // Prepare output chunk with same schema as input
    
    SelectionVector sel_vector(STANDARD_VECTOR_SIZE);
    idx_t output_count = 0;

    // Process each row in the input chunk
    for (idx_t row_idx = 0; row_idx < input.size(); row_idx++) {
        // Get the physical index for this row
        idx_t i_url = url_uvf.sel->get_index(row_idx);
        idx_t i_event_time = event_time_uvf.sel->get_index(row_idx);
        
        // Check if URL is valid (not NULL)
        if (!url_uvf.validity.RowIsValid(i_url)) {
            continue; // Skip NULL URLs
        }
        
        string_t url_val = url_ptr[i_url];
        timestamp_t event_time_val = event_time_ptr[i_event_time];
        
        // Apply filter: URL LIKE '%google%'
        std::string url_str = url_val.GetString();
        if (url_str.find("google") != std::string::npos) {
            // Row passes the filter, add to selection
            sel_vector.set_index(output_count, row_idx);
            output_count++;
            
            // Store for sorting - lock when accessing global state
            {
                std::lock_guard<std::mutex> guard(g.lock);
                g.sort_state.AddRow(event_time_val, g.sort_state.chunks.size(), output_count - 1);
            }
        }
    }
    
    // Set the cardinality of the output chunk
    output.SetCardinality(output_count);
    
    // Copy selected rows to output
    if (output_count > 0) {
        for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
            output.data[col_idx].Slice(input.data[col_idx], sel_vector, output_count);
        }
        
        // Store the chunk for later access during finalize
        {
            std::lock_guard<std::mutex> guard(g.lock);
            g.sort_state.chunks.push_back(DataChunk());
            g.sort_state.chunks.back().Initialize(Allocator::GetDefaultAllocator(), input.GetTypes());
            g.sort_state.chunks.back().Reference(output);
        }
    }

    return OperatorResultType::HAVE_MORE_OUTPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();
    // Merge the local state into the global state exactly once.
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            //TODO: merge local state with global state
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    // Only the *last* local state to merge emits the final result.
    // All other threads return FINISHED with an empty chunk.
    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            // Sort all collected rows by EventTime in ascending order
            g.sort_state.SortNow();
        }
        
        // Populate out chunk with sorted results
        idx_t output_row = 0;
        for (size_t i = 0; i < g.sort_state.buffer.size(); ++i) {
            if (output_row >= STANDARD_VECTOR_SIZE) {
                break; // Only fill one output chunk at a time
            }
            
            idx_t chunk_idx = 0; // All data was stored in a single chunk during execution
            idx_t row_idx = g.sort_state.row_indices[chunk_idx].get_index(i);
            
            // Copy row data from the stored chunk to output
            for (idx_t col_idx = 0; col_idx < g.sort_state.chunks[chunk_idx].ColumnCount(); col_idx++) {
                out.data[col_idx].SetValue(output_row, g.sort_state.chunks[chunk_idx].data[col_idx].GetValue(row_idx));
            }
            output_row++;
        }
        out.SetCardinality(output_row);
    } else {
        out.SetCardinality(0);
    }
    
    return OperatorFinalizeResultType::FINISHED;
}

static void LoadInternal(ExtensionLoader &loader) {
    TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
    f.in_out_function       = FnExecute;
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

} 
extern "C" {
    DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) {
        duckdb::LoadInternal(loader);
    }
}