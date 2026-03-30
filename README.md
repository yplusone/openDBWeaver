# openDBWeaver

An experimental pipeline for **custom query execution in DuckDB (C++)**: large language models generate sketch code from SQL workloads, you compile and validate it in Docker, then iteratively improve performance with **tree search and hint-guided optimization**.


## Requirements

- **Python** 3.10+ (recommended)
- **Docker** (to compile the generated C++ inside the image and run the DuckDB CLI)
- A local **DuckDB database file** (`.duckdb`) and paths to your **DuckDB / build tree** used to compile the execution code, aligned with `DB_PATH`, `DEFAULT_SOURCE_DIR`, `DUCKDB_BINARY_PATH`, etc. in `config.py`
- A reachable **OpenAI-compatible API** (via `langchain-openai`) for sketch, fix, and optimization stages

## Docker Setup

A `Dockerfile` is already provided in this repository, so you only need to build the image and run the container to compile the project.

### 1. Build the Docker image

```bash
docker build \
  --network=host \
  --build-arg GIT_HTTP_PROXY=http://127.0.0.1:31328 \
  --build-arg GIT_HTTPS_PROXY=http://127.0.0.1:31328 \
  -t dbweaver_duckdb:latest .
```
This command builds the Docker image dbweaver_duckdb:latest using the provided Dockerfile.

If your environment does not require a proxy, you can remove the two --build-arg options.

### 2.Build the project inside the container

```
docker run --rm   \
    -v "extension-template/:/app"   \
    -w /app   \
    --network=host   \
    -e BUILD_TYPE=release   \
    -e MAKEFLAGS=-j   \
    dbweaver_duckdb:latest   \
    bash -lc 'rm -rf /app/build/release && make release'
```

## Benchmarks and workloads


For ClickBench, Follow the instruction in https://github.com/ClickHouse/ClickBench

For SSB, Follow the instruction in https://clickhouse.com/docs/getting-started/example-datasets/star-schema

When switching benchmarks, update `BENCHMARK`, `GENERATED_CODE_DIR`and related paths in `config.py` together.


## Installing dependencies

```bash
pip install httpx langgraph langchain-core langchain-openai \
  pandas numpy sqlparse sqlglot pydantic typing-extensions
```

Run scripts from the repository root so `import config` and `dbweaver` resolve correctly.

## Configuration

Copy and edit `config_example.py` as `config.py` and verify at least:

- **Benchmark**: `BENCHMARK` (e.g. `hits` / `ssb`)
- **Data and binaries**: `DB_PATH`, `DUCKDB_BINARY_PATH`, `DEFAULT_SOURCE_DIR`, `DOCKER_IMAGE`.
- **Output directories**: `GENERATED_CODE_DIR`.
- **LLM**: `API_KEY`, `BASE_URL`, per-stage `*_MODEL`, etc. **Do not commit real API keys.**
```
export DBWEAVER_API_KEY="your key"
export DBWEAVER_BASE_URL="https://api.openai.com/v1"
```

## Typical usage

All commands below assume the project root as the current working directory.

### 1. Generate sketch code

```bash
python scripts/incremental_sketch_construction.py
```

The script iterates over the configured `query_id` values and writes outputs under `SKETCH_DIR` (see `config.py`).

### 2. Compile / result fix

```bash
python scripts/fix_code.py
```


### 3. Hint-guided performance optimization

```bash
python scripts/hint_guided_code_optimization.py
```

Expects sources already present under `SKETCH_FIX_DIR`; streams the optimization graph and writes tree-search traces under `TRACE_DIR`.


## Notes

- Paths in `config.py` (local DuckDB files, Docker volume mounts) are machine-specific; re-check them when moving to another host.
- Generation and optimization call external APIs and may keep Docker / CPU busy for a long time; they are best suited to experimental environments.
