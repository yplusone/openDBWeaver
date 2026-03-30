#define DUCKDB_EXTENSION_MAIN

//<<HEADERS>>

namespace duckdb {



//<<SUPPORT_CODE>>

//<<BIND_CODE>>

//<<EXECUTE_CODE>>

//<<FINALIZE_CODE>>

//<<FUNCTION_DEFINE>>

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

