import httpx
import os

# Optional: set HTTP_PROXY / HTTPS_PROXY in the environment if you need a proxy.

BENCHMARK = "hits"
THREADS = 8
DB_PATH = os.environ.get("DB_PATH", "/path/to/your/database.duckdb")

DUCKDB_BINARY_PATH = os.environ.get("DUCKDB_BINARY_PATH", "/path/to/duckdb/extension-template/build/")
DEFAULT_SOURCE_DIR = os.environ.get("DEFAULT_SOURCE_DIR", "/path/to/duckdb/extension-template/")
DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "dbweaver_duckdb:latest")

GENERATED_CODE_DIR = f"output/{BENCHMARK}/gemini_3_flash"
SKETCH_DIR = f"{GENERATED_CODE_DIR}/sketch"
SKETCH_FIX_DIR = f"{GENERATED_CODE_DIR}/sketch_fix"
OPTIMIZED_CODE_DIR = f"{GENERATED_CODE_DIR}/optimized_code"
TRACE_DIR = f"{GENERATED_CODE_DIR}/trace/"


MAX_ITERATIONS_PER_HINT = 10

_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
client = httpx.Client(
    verify=False,
    http2=False,
    timeout=httpx.Timeout(
        connect=5.0,
        read=300.0,
        write=30.0,
        pool=5.0,
    ), 
    proxy=_proxy or None,
)


EXPT_LLM_MODEL = "gpt-4o-mini"
OPTIMIZE_LLM_MODEL = "gpt-5"
CODEGEN_LLM_MODEL = "gemini-3-flash-preview"
API_KEY = os.environ.get("DBWEAVER_API_KEY", "")
BASE_URL = os.environ.get("DBWEAVER_BASE_URL", "https://api.openai.com/v1")
CODEGEN_API_KEY = API_KEY
CODEGEN_BASE_URL = BASE_URL

# Validation flags
VALIDATION_FLAG_COMPILE_FAIL = "COMPILE_FAIL"
VALIDATION_FLAG_RESULT_MISMATCH = "RESULT_MISMATCH"
VALIDATION_FLAG_PERF_NEED_OPT = "PERF_NEED_OPT"


TEMPLATE_PATH = "template/one_input_parralel.cpp"
