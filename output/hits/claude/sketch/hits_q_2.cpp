/*
query_template: SELECT COUNT(*) AS cnt FROM hits WHERE AdvEngineID <> 0;

split_template: select * from dbweaver((SELECT * FROM hits WHERE (AdvEngineID!=0)));
query_example: SELECT COUNT(*) AS cnt FROM hits WHERE AdvEngineID <> 0;

split_query: select * from dbweaver((SELECT * FROM hits WHERE (AdvEngineID!=0)));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
    
    // Accumulator for count_star
    uint64_t cnt = 0;
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;
    
    // Accumulator for count_star
    uint64_t cnt = 0;
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
    // Define output schema: cnt as BIGINT
    return_types.push_back(LogicalType::BIGINT);
    names.push_back("cnt");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat WatchID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), WatchID_uvf);
    int64_t* WatchID_ptr = (int64_t*)WatchID_uvf.data;
    
    UnifiedVectorFormat JavaEnable_uvf;
    input.data[1].ToUnifiedFormat(input.size(), JavaEnable_uvf);
    int16_t* JavaEnable_ptr = (int16_t*)JavaEnable_uvf.data;
    
    UnifiedVectorFormat Title_uvf;
    input.data[2].ToUnifiedFormat(input.size(), Title_uvf);
    string_t* Title_ptr = (string_t*)Title_uvf.data;
    
    UnifiedVectorFormat GoodEvent_uvf;
    input.data[3].ToUnifiedFormat(input.size(), GoodEvent_uvf);
    int16_t* GoodEvent_ptr = (int16_t*)GoodEvent_uvf.data;
    
    UnifiedVectorFormat EventTime_uvf;
    input.data[4].ToUnifiedFormat(input.size(), EventTime_uvf);
    timestamp_t* EventTime_ptr = (timestamp_t*)EventTime_uvf.data;
    
    UnifiedVectorFormat EventDate_uvf;
    input.data[5].ToUnifiedFormat(input.size(), EventDate_uvf);
    date_t* EventDate_ptr = (date_t*)EventDate_uvf.data;
    
    UnifiedVectorFormat CounterID_uvf;
    input.data[6].ToUnifiedFormat(input.size(), CounterID_uvf);
    int32_t* CounterID_ptr = (int32_t*)CounterID_uvf.data;
    
    UnifiedVectorFormat ClientIP_uvf;
    input.data[7].ToUnifiedFormat(input.size(), ClientIP_uvf);
    int32_t* ClientIP_ptr = (int32_t*)ClientIP_uvf.data;
    
    UnifiedVectorFormat RegionID_uvf;
    input.data[8].ToUnifiedFormat(input.size(), RegionID_uvf);
    int32_t* RegionID_ptr = (int32_t*)RegionID_uvf.data;
    
    UnifiedVectorFormat UserID_uvf;
    input.data[9].ToUnifiedFormat(input.size(), UserID_uvf);
    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;
    
    UnifiedVectorFormat CounterClass_uvf;
    input.data[10].ToUnifiedFormat(input.size(), CounterClass_uvf);
    int16_t* CounterClass_ptr = (int16_t*)CounterClass_uvf.data;
    
    UnifiedVectorFormat OS_uvf;
    input.data[11].ToUnifiedFormat(input.size(), OS_uvf);
    int16_t* OS_ptr = (int16_t*)OS_uvf.data;
    
    UnifiedVectorFormat UserAgent_uvf;
    input.data[12].ToUnifiedFormat(input.size(), UserAgent_uvf);
    int16_t* UserAgent_ptr = (int16_t*)UserAgent_uvf.data;
    
    UnifiedVectorFormat URL_uvf;
    input.data[13].ToUnifiedFormat(input.size(), URL_uvf);
    string_t* URL_ptr = (string_t*)URL_uvf.data;
    
    UnifiedVectorFormat ParamCurrencyID_uvf;
    input.data[14].ToUnifiedFormat(input.size(), ParamCurrencyID_uvf);
    int16_t* ParamCurrencyID_ptr = (int16_t*)ParamCurrencyID_uvf.data;
    
    UnifiedVectorFormat OpenstatServiceName_uvf;
    input.data[15].ToUnifiedFormat(input.size(), OpenstatServiceName_uvf);
    string_t* OpenstatServiceName_ptr = (string_t*)OpenstatServiceName_uvf.data;
    
    UnifiedVectorFormat OpenstatCampaignID_uvf;
    input.data[16].ToUnifiedFormat(input.size(), OpenstatCampaignID_uvf);
    string_t* OpenstatCampaignID_ptr = (string_t*)OpenstatCampaignID_uvf.data;
    
    UnifiedVectorFormat OpenstatAdID_uvf;
    input.data[17].ToUnifiedFormat(input.size(), OpenstatAdID_uvf);
    string_t* OpenstatAdID_ptr = (string_t*)OpenstatAdID_uvf.data;
    
    UnifiedVectorFormat OpenstatSourceID_uvf;
    input.data[18].ToUnifiedFormat(input.size(), OpenstatSourceID_uvf);
    string_t* OpenstatSourceID_ptr = (string_t*)OpenstatSourceID_uvf.data;
    
    UnifiedVectorFormat UTMSource_uvf;
    input.data[19].ToUnifiedFormat(input.size(), UTMSource_uvf);
    string_t* UTMSource_ptr = (string_t*)UTMSource_uvf.data;
    
    UnifiedVectorFormat UTMMedium_uvf;
    input.data[20].ToUnifiedFormat(input.size(), UTMMedium_uvf);
    string_t* UTMMedium_ptr = (string_t*)UTMMedium_uvf.data;
    
    UnifiedVectorFormat UTMCampaign_uvf;
    input.data[21].ToUnifiedFormat(input.size(), UTMCampaign_uvf);
    string_t* UTMCampaign_ptr = (string_t*)UTMCampaign_uvf.data;
    
    UnifiedVectorFormat UTMContent_uvf;
    input.data[22].ToUnifiedFormat(input.size(), UTMContent_uvf);
    string_t* UTMContent_ptr = (string_t*)UTMContent_uvf.data;
    
    UnifiedVectorFormat UTMTerm_uvf;
    input.data[23].ToUnifiedFormat(input.size(), UTMTerm_uvf);
    string_t* UTMTerm_ptr = (string_t*)UTMTerm_uvf.data;
    
    UnifiedVectorFormat FromTag_uvf;
    input.data[24].ToUnifiedFormat(input.size(), FromTag_uvf);
    string_t* FromTag_ptr = (string_t*)FromTag_uvf.data;
    
    UnifiedVectorFormat HasGCLID_uvf;
    input.data[25].ToUnifiedFormat(input.size(), HasGCLID_uvf);
    int16_t* HasGCLID_ptr = (int16_t*)HasGCLID_uvf.data;
    
    UnifiedVectorFormat RefererHash_uvf;
    input.data[26].ToUnifiedFormat(input.size(), RefererHash_uvf);
    int64_t* RefererHash_ptr = (int64_t*)RefererHash_uvf.data;
    
    UnifiedVectorFormat URLHash_uvf;
    input.data[27].ToUnifiedFormat(input.size(), URLHash_uvf);
    int64_t* URLHash_ptr = (int64_t*)URLHash_uvf.data;
    
    UnifiedVectorFormat CLID_uvf;
    input.data[28].ToUnifiedFormat(input.size(), CLID_uvf);
    int32_t* CLID_ptr = (int32_t*)CLID_uvf.data;
    
    // validity bitmaps
    auto &valid_WatchID  = WatchID_uvf.validity;
    auto &valid_JavaEnable  = JavaEnable_uvf.validity;
    auto &valid_Title  = Title_uvf.validity;
    auto &valid_GoodEvent  = GoodEvent_uvf.validity;
    auto &valid_EventTime  = EventTime_uvf.validity;
    auto &valid_EventDate  = EventDate_uvf.validity;
    auto &valid_CounterID  = CounterID_uvf.validity;
    auto &valid_ClientIP  = ClientIP_uvf.validity;
    auto &valid_RegionID  = RegionID_uvf.validity;
    auto &valid_UserID  = UserID_uvf.validity;
    auto &valid_CounterClass  = CounterClass_uvf.validity;
    auto &valid_OS  = OS_uvf.validity;
    auto &valid_UserAgent  = UserAgent_uvf.validity;
    auto &valid_URL  = URL_uvf.validity;
    auto &valid_ParamCurrencyID  = ParamCurrencyID_uvf.validity;
    auto &valid_OpenstatServiceName  = OpenstatServiceName_uvf.validity;
    auto &valid_OpenstatCampaignID  = OpenstatCampaignID_uvf.validity;
    auto &valid_OpenstatAdID  = OpenstatAdID_uvf.validity;
    auto &valid_OpenstatSourceID  = OpenstatSourceID_uvf.validity;
    auto &valid_UTMSource  = UTMSource_uvf.validity;
    auto &valid_UTMMedium  = UTMMedium_uvf.validity;
    auto &valid_UTMCampaign  = UTMCampaign_uvf.validity;
    auto &valid_UTMContent  = UTMContent_uvf.validity;
    auto &valid_UTMTerm  = UTMTerm_uvf.validity;
    auto &valid_FromTag  = FromTag_uvf.validity;
    auto &valid_HasGCLID  = HasGCLID_uvf.validity;
    auto &valid_RefererHash  = RefererHash_uvf.validity;
    auto &valid_URLHash  = URLHash_uvf.validity;
    auto &valid_CLID  = CLID_uvf.validity;
    
    const bool WatchID_all_valid = valid_WatchID.AllValid();
    const bool JavaEnable_all_valid = valid_JavaEnable.AllValid();
    const bool Title_all_valid = valid_Title.AllValid();
    const bool GoodEvent_all_valid = valid_GoodEvent.AllValid();
    const bool EventTime_all_valid = valid_EventTime.AllValid();
    const bool EventDate_all_valid = valid_EventDate.AllValid();
    const bool CounterID_all_valid = valid_CounterID.AllValid();
    const bool ClientIP_all_valid = valid_ClientIP.AllValid();
    const bool RegionID_all_valid = valid_RegionID.AllValid();
    const bool UserID_all_valid = valid_UserID.AllValid();
    const bool CounterClass_all_valid = valid_CounterClass.AllValid();
    const bool OS_all_valid = valid_OS.AllValid();
    const bool UserAgent_all_valid = valid_UserAgent.AllValid();
    const bool URL_all_valid = valid_URL.AllValid();
    const bool ParamCurrencyID_all_valid = valid_ParamCurrencyID.AllValid();
    const bool OpenstatServiceName_all_valid = valid_OpenstatServiceName.AllValid();
    const bool OpenstatCampaignID_all_valid = valid_OpenstatCampaignID.AllValid();
    const bool OpenstatAdID_all_valid = valid_OpenstatAdID.AllValid();
    const bool OpenstatSourceID_all_valid = valid_OpenstatSourceID.AllValid();
    const bool UTMSource_all_valid = valid_UTMSource.AllValid();
    const bool UTMMedium_all_valid = valid_UTMMedium.AllValid();
    const bool UTMCampaign_all_valid = valid_UTMCampaign.AllValid();
    const bool UTMContent_all_valid = valid_UTMContent.AllValid();
    const bool UTMTerm_all_valid = valid_UTMTerm.AllValid();
    const bool FromTag_all_valid = valid_FromTag.AllValid();
    const bool HasGCLID_all_valid = valid_HasGCLID.AllValid();
    const bool RefererHash_all_valid = valid_RefererHash.AllValid();
    const bool URLHash_all_valid = valid_URLHash.AllValid();
    const bool CLID_all_valid = valid_CLID.AllValid();
    
    // Process rows with null handling
    if (WatchID_all_valid && JavaEnable_all_valid && Title_all_valid && GoodEvent_all_valid && EventTime_all_valid && EventDate_all_valid && CounterID_all_valid && ClientIP_all_valid && RegionID_all_valid && UserID_all_valid && CounterClass_all_valid && OS_all_valid && UserAgent_all_valid && URL_all_valid && ParamCurrencyID_all_valid && OpenstatServiceName_all_valid && OpenstatCampaignID_all_valid && OpenstatAdID_all_valid && OpenstatSourceID_all_valid && UTMSource_all_valid && UTMMedium_all_valid && UTMCampaign_all_valid && UTMContent_all_valid && UTMTerm_all_valid && FromTag_all_valid && HasGCLID_all_valid && RefererHash_all_valid && URLHash_all_valid && CLID_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_WatchID = WatchID_uvf.sel->get_index(row_idx);
            idx_t i_JavaEnable = JavaEnable_uvf.sel->get_index(row_idx);
            idx_t i_Title = Title_uvf.sel->get_index(row_idx);
            idx_t i_GoodEvent = GoodEvent_uvf.sel->get_index(row_idx);
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);
            idx_t i_EventDate = EventDate_uvf.sel->get_index(row_idx);
            idx_t i_CounterID = CounterID_uvf.sel->get_index(row_idx);
            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);
            idx_t i_RegionID = RegionID_uvf.sel->get_index(row_idx);
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            idx_t i_CounterClass = CounterClass_uvf.sel->get_index(row_idx);
            idx_t i_OS = OS_uvf.sel->get_index(row_idx);
            idx_t i_UserAgent = UserAgent_uvf.sel->get_index(row_idx);
            idx_t i_URL = URL_uvf.sel->get_index(row_idx);
            idx_t i_ParamCurrencyID = ParamCurrencyID_uvf.sel->get_index(row_idx);
            idx_t i_OpenstatServiceName = OpenstatServiceName_uvf.sel->get_index(row_idx);
            idx_t i_OpenstatCampaignID = OpenstatCampaignID_uvf.sel->get_index(row_idx);
            idx_t i_OpenstatAdID = OpenstatAdID_uvf.sel->get_index(row_idx);
            idx_t i_OpenstatSourceID = OpenstatSourceID_uvf.sel->get_index(row_idx);
            idx_t i_UTMSource = UTMSource_uvf.sel->get_index(row_idx);
            idx_t i_UTMMedium = UTMMedium_uvf.sel->get_index(row_idx);
            idx_t i_UTMCampaign = UTMCampaign_uvf.sel->get_index(row_idx);
            idx_t i_UTMContent = UTMContent_uvf.sel->get_index(row_idx);
            idx_t i_UTMTerm = UTMTerm_uvf.sel->get_index(row_idx);
            idx_t i_FromTag = FromTag_uvf.sel->get_index(row_idx);
            idx_t i_HasGCLID = HasGCLID_uvf.sel->get_index(row_idx);
            idx_t i_RefererHash = RefererHash_uvf.sel->get_index(row_idx);
            idx_t i_URLHash = URLHash_uvf.sel->get_index(row_idx);
            idx_t i_CLID = CLID_uvf.sel->get_index(row_idx);

            int64_t v_WatchID = WatchID_ptr[i_WatchID];
            int16_t v_JavaEnable = JavaEnable_ptr[i_JavaEnable];
            string_t v_Title = Title_ptr[i_Title];
            int16_t v_GoodEvent = GoodEvent_ptr[i_GoodEvent];
            timestamp_t v_EventTime = EventTime_ptr[i_EventTime];
            date_t v_EventDate = EventDate_ptr[i_EventDate];
            int32_t v_CounterID = CounterID_ptr[i_CounterID];
            int32_t v_ClientIP = ClientIP_ptr[i_ClientIP];
            int32_t v_RegionID = RegionID_ptr[i_RegionID];
            int64_t v_UserID = UserID_ptr[i_UserID];
            int16_t v_CounterClass = CounterClass_ptr[i_CounterClass];
            int16_t v_OS = OS_ptr[i_OS];
            int16_t v_UserAgent = UserAgent_ptr[i_UserAgent];
            string_t v_URL = URL_ptr[i_URL];
            int16_t v_ParamCurrencyID = ParamCurrencyID_ptr[i_ParamCurrencyID];
            string_t v_OpenstatServiceName = OpenstatServiceName_ptr[i_OpenstatServiceName];
            string_t v_OpenstatCampaignID = OpenstatCampaignID_ptr[i_OpenstatCampaignID];
            string_t v_OpenstatAdID = OpenstatAdID_ptr[i_OpenstatAdID];
            string_t v_OpenstatSourceID = OpenstatSourceID_ptr[i_OpenstatSourceID];
            string_t v_UTMSource = UTMSource_ptr[i_UTMSource];
            string_t v_UTMMedium = UTMMedium_ptr[i_UTMMedium];
            string_t v_UTMCampaign = UTMCampaign_ptr[i_UTMCampaign];
            string_t v_UTMContent = UTMContent_ptr[i_UTMContent];
            string_t v_UTMTerm = UTMTerm_ptr[i_UTMTerm];
            string_t v_FromTag = FromTag_ptr[i_FromTag];
            int16_t v_HasGCLID = HasGCLID_ptr[i_HasGCLID];
            int64_t v_RefererHash = RefererHash_ptr[i_RefererHash];
            int64_t v_URLHash = URLHash_ptr[i_URLHash];
            int32_t v_CLID = CLID_ptr[i_CLID];

            // ======================================
            //  Core computation logic (no NULLs)
            // Increment count for count_star aggregate
            l.cnt++;
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_WatchID = WatchID_uvf.sel->get_index(row_idx);
            idx_t i_JavaEnable = JavaEnable_uvf.sel->get_index(row_idx);
            idx_t i_Title = Title_uvf.sel->get_index(row_idx);
            idx_t i_GoodEvent = GoodEvent_uvf.sel->get_index(row_idx);
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);
            idx_t i_EventDate = EventDate_uvf.sel->get_index(row_idx);
            idx_t i_CounterID = CounterID_uvf.sel->get_index(row_idx);
            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);
            idx_t i_RegionID = RegionID_uvf.sel->get_index(row_idx);
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            idx_t i_CounterClass = CounterClass_uvf.sel->get_index(row_idx);
            idx_t i_OS = OS_uvf.sel->get_index(row_idx);
            idx_t i_UserAgent = UserAgent_uvf.sel->get_index(row_idx);
            idx_t i_URL = URL_uvf.sel->get_index(row_idx);
            idx_t i_ParamCurrencyID = ParamCurrencyID_uvf.sel->get_index(row_idx);
            idx_t i_OpenstatServiceName = OpenstatServiceName_uvf.sel->get_index(row_idx);
            idx_t i_OpenstatCampaignID = OpenstatCampaignID_uvf.sel->get_index(row_idx);
            idx_t i_OpenstatAdID = OpenstatAdID_uvf.sel->get_index(row_idx);
            idx_t i_OpenstatSourceID = OpenstatSourceID_uvf.sel->get_index(row_idx);
            idx_t i_UTMSource = UTMSource_uvf.sel->get_index(row_idx);
            idx_t i_UTMMedium = UTMMedium_uvf.sel->get_index(row_idx);
            idx_t i_UTMCampaign = UTMCampaign_uvf.sel->get_index(row_idx);
            idx_t i_UTMContent = UTMContent_uvf.sel->get_index(row_idx);
            idx_t i_UTMTerm = UTMTerm_uvf.sel->get_index(row_idx);
            idx_t i_FromTag = FromTag_uvf.sel->get_index(row_idx);
            idx_t i_HasGCLID = HasGCLID_uvf.sel->get_index(row_idx);
            idx_t i_RefererHash = RefererHash_uvf.sel->get_index(row_idx);
            idx_t i_URLHash = URLHash_uvf.sel->get_index(row_idx);
            idx_t i_CLID = CLID_uvf.sel->get_index(row_idx);

            if (!WatchID_all_valid && !valid_WatchID.RowIsValid(i_WatchID)) {
                continue; // row is NULL in column 1 → skip
            }
            if (!JavaEnable_all_valid && !valid_JavaEnable.RowIsValid(i_JavaEnable)) {
                continue;
            }
            if (!Title_all_valid && !valid_Title.RowIsValid(i_Title)) {
                continue;
            }
            if (!GoodEvent_all_valid && !valid_GoodEvent.RowIsValid(i_GoodEvent)) {
                continue;
            }
            if (!EventTime_all_valid && !valid_EventTime.RowIsValid(i_EventTime)) {
                continue;
            }
            if (!EventDate_all_valid && !valid_EventDate.RowIsValid(i_EventDate)) {
                continue;
            }
            if (!CounterID_all_valid && !valid_CounterID.RowIsValid(i_CounterID)) {
                continue;
            }
            if (!ClientIP_all_valid && !valid_ClientIP.RowIsValid(i_ClientIP)) {
                continue;
            }
            if (!RegionID_all_valid && !valid_RegionID.RowIsValid(i_RegionID)) {
                continue;
            }
            if (!UserID_all_valid && !valid_UserID.RowIsValid(i_UserID)) {
                continue;
            }
            if (!CounterClass_all_valid && !valid_CounterClass.RowIsValid(i_CounterClass)) {
                continue;
            }
            if (!OS_all_valid && !valid_OS.RowIsValid(i_OS)) {
                continue;
            }
            if (!UserAgent_all_valid && !valid_UserAgent.RowIsValid(i_UserAgent)) {
                continue;
            }
            if (!URL_all_valid && !valid_URL.RowIsValid(i_URL)) {
                continue;
            }
            if (!ParamCurrencyID_all_valid && !valid_ParamCurrencyID.RowIsValid(i_ParamCurrencyID)) {
                continue;
            }
            if (!OpenstatServiceName_all_valid && !valid_OpenstatServiceName.RowIsValid(i_OpenstatServiceName)) {
                continue;
            }
            if (!OpenstatCampaignID_all_valid && !valid_OpenstatCampaignID.RowIsValid(i_OpenstatCampaignID)) {
                continue;
            }
            if (!OpenstatAdID_all_valid && !valid_OpenstatAdID.RowIsValid(i_OpenstatAdID)) {
                continue;
            }
            if (!OpenstatSourceID_all_valid && !valid_OpenstatSourceID.RowIsValid(i_OpenstatSourceID)) {
                continue;
            }
            if (!UTMSource_all_valid && !valid_UTMSource.RowIsValid(i_UTMSource)) {
                continue;
            }
            if (!UTMMedium_all_valid && !valid_UTMMedium.RowIsValid(i_UTMMedium)) {
                continue;
            }
            if (!UTMCampaign_all_valid && !valid_UTMCampaign.RowIsValid(i_UTMCampaign)) {
                continue;
            }
            if (!UTMContent_all_valid && !valid_UTMContent.RowIsValid(i_UTMContent)) {
                continue;
            }
            if (!UTMTerm_all_valid && !valid_UTMTerm.RowIsValid(i_UTMTerm)) {
                continue;
            }
            if (!FromTag_all_valid && !valid_FromTag.RowIsValid(i_FromTag)) {
                continue;
            }
            if (!HasGCLID_all_valid && !valid_HasGCLID.RowIsValid(i_HasGCLID)) {
                continue;
            }
            if (!RefererHash_all_valid && !valid_RefererHash.RowIsValid(i_RefererHash)) {
                continue;
            }
            if (!URLHash_all_valid && !valid_URLHash.RowIsValid(i_URLHash)) {
                continue;
            }
            if (!CLID_all_valid && !valid_CLID.RowIsValid(i_CLID)) {
                continue;
            }

            // At this point, all required columns are valid for this row

            int64_t v_WatchID = WatchID_ptr[i_WatchID];
            int16_t v_JavaEnable = JavaEnable_ptr[i_JavaEnable];
            string_t v_Title = Title_ptr[i_Title];
            int16_t v_GoodEvent = GoodEvent_ptr[i_GoodEvent];
            timestamp_t v_EventTime = EventTime_ptr[i_EventTime];
            date_t v_EventDate = EventDate_ptr[i_EventDate];
            int32_t v_CounterID = CounterID_ptr[i_CounterID];
            int32_t v_ClientIP = ClientIP_ptr[i_ClientIP];
            int32_t v_RegionID = RegionID_ptr[i_RegionID];
            int64_t v_UserID = UserID_ptr[i_UserID];
            int16_t v_CounterClass = CounterClass_ptr[i_CounterClass];
            int16_t v_OS = OS_ptr[i_OS];
            int16_t v_UserAgent = UserAgent_ptr[i_UserAgent];
            string_t v_URL = URL_ptr[i_URL];
            int16_t v_ParamCurrencyID = ParamCurrencyID_ptr[i_ParamCurrencyID];
            string_t v_OpenstatServiceName = OpenstatServiceName_ptr[i_OpenstatServiceName];
            string_t v_OpenstatCampaignID = OpenstatCampaignID_ptr[i_OpenstatCampaignID];
            string_t v_OpenstatAdID = OpenstatAdID_ptr[i_OpenstatAdID];
            string_t v_OpenstatSourceID = OpenstatSourceID_ptr[i_OpenstatSourceID];
            string_t v_UTMSource = UTMSource_ptr[i_UTMSource];
            string_t v_UTMMedium = UTMMedium_ptr[i_UTMMedium];
            string_t v_UTMCampaign = UTMCampaign_ptr[i_UTMCampaign];
            string_t v_UTMContent = UTMContent_ptr[i_UTMContent];
            string_t v_UTMTerm = UTMTerm_ptr[i_UTMTerm];
            string_t v_FromTag = FromTag_ptr[i_FromTag];
            int16_t v_HasGCLID = HasGCLID_ptr[i_HasGCLID];
            int64_t v_RefererHash = RefererHash_ptr[i_RefererHash];
            int64_t v_URLHash = URLHash_ptr[i_URLHash];
            int32_t v_CLID = CLID_ptr[i_CLID];

            // ======================================
            //  Core computation logic (NULL-safe)
            // Increment count for count_star aggregate
            l.cnt++;
            // ======================================
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();
    // Merge the local state into the global state exactly once.
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            g.cnt += l.cnt;
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    // Only the *last* local state to merge emits the final result.
    // All other threads return FINISHED with an empty chunk.
    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        uint64_t final_cnt;
        {
            std::lock_guard<std::mutex> guard(g.lock);
            final_cnt = g.cnt;
        }
        out.SetCardinality(1);
        out.SetValue(0, 0, Value::UBIGINT(final_cnt));
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