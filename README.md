# openDBWeaver

An experimental pipeline for **custom query execution in DuckDB (C++)**: large language models generate sketch code from SQL workloads, you compile and validate it in Docker, then iteratively improve performance with **tree search and hint-guided optimization**.

## Features

- **Sketch generation**: A `LangGraph` subgraph chains **plan → per-operator codegen → merge** and emits single-file C++ sources that implement the query execution (see `template/` and `dbweaver/sketch/`).
- **Automatic fix**: Iteratively calls an LLM on compile failures, result mismatches, and similar issues; supports patch vs full-replacement style outputs (`dbweaver/optimize/fix_agent.py`).
- **Hint-guided optimization**: After validation passes, explores rewrites with a search tree and records traces and performance (`dbweaver/optimize/graph.py`, etc.).
- **Execution and validation**: `CodeChecker` uses `DockerDuckDBRunner` / `DuckDBConnector` for builds, query runs, and result comparison (`dbweaver/env/`).

## Repository layout (selected)

| Path | Description |
|------|-------------|
| `config.py` | Global paths, benchmark name, LLM settings, output dirs (configure locally; do not commit secrets) |
| `dbweaver/sketch/` | Sketch generation, operator codegen, `GatherContext` |
| `dbweaver/optimize/` | Optimization graph, candidate chains, scoring, `fix_agent` |
| `dbweaver/env/` | Docker builds, DuckDB execution, result comparison |
| `benchmark/` | Workloads (e.g. `hits_queries.sql`, `ssb_benchmark`) and query loaders |
| `scripts/` | Runnable entry-point scripts |
| `output/` | Generated artifacts (sketches, fixed code, optimized code, traces, etc.; paths come from `config.py`) |

## Requirements

- **Python** 3.10+ (recommended)
- **Docker** (to compile the generated C++ inside the image and run the DuckDB CLI)
- A local **DuckDB database file** (`.duckdb`) and paths to your **DuckDB / build tree** used to compile the execution code, aligned with `DB_PATH`, `DEFAULT_SOURCE_DIR`, `DUCKDB_BINARY_PATH`, etc. in `config.py`
- A reachable **OpenAI-compatible API** (via `langchain-openai`) for sketch, fix, and optimization stages

## Docker establish
```
docker build \
  --network=host \
  --build-arg GIT_HTTP_PROXY=http://127.0.0.1:31328 \
  --build-arg GIT_HTTPS_PROXY=http://127.0.0.1:31328 \
  -t dbweaver_duckdb:latest .
```

Optional: You can make through docker keep environment clean
```
docker run --rm   \
    -v "/home/yjn/duckdb_docker/extension-template/:/app"   \
    -w /app   \
    --network=host   \
    -e BUILD_TYPE=release   \
    -e MAKEFLAGS=-j   \
    dbweaver_duckdb:latest   \
    bash -lc 'rm -rf /app/build/release && make release'
```

## Installing dependencies

There is no bundled `requirements.txt`; install from imports as needed, for example:

```bash
pip install httpx langgraph langchain-core langchain-openai \
  pandas numpy sqlparse sqlglot pydantic typing-extensions
```

Run scripts from the repository root so `import config` and `dbweaver` resolve correctly.

## Configuration

Copy and edit `config.py` (or keep a private local copy) and verify at least:

- **Benchmark**: `BENCHMARK` (e.g. `hits` / `ssb`) and that the matching `*_database.json` under `benchmark/profiles/` exists.
- **Data and binaries**: `DB_PATH`, `DUCKDB_BINARY_PATH`, `DEFAULT_SOURCE_DIR`, `DOCKER_IMAGE`, `THREADS`.
- **Output directories**: `GENERATED_CODE_DIR`, `SKETCH_DIR`, `SKETCH_FIX_DIR`, `OPTIMIZED_CODE_DIR`, `TRACE_DIR`.
- **LLM**: `API_KEY`, `BASE_URL`, per-stage `*_MODEL`, etc. **Do not commit real API keys.**
- **Networking**: If you use a proxy, configure `httpx` `proxy` and environment variables to match your setup.

## Typical usage

All commands below assume the project root as the current working directory.

### 1. Generate sketch code

```bash
python scripts/incremental_sketch_construction.py
```

The script iterates over the configured `query_id` values and writes outputs under `SKETCH_DIR` (see `config.py`).

### 2. Compile / result fix

```bash
python dbweaver/optimize/fix_agent.py
```

By default it reads the `.cpp` for selected queries from `SKETCH_DIR`, writes successful fixes to `SKETCH_FIX_DIR`, and may append rows to `fix_agent_iterations.csv`.

### 3. Hint-guided performance optimization

```bash
python scripts/hint_guided_code_optimization.py
```

Expects sources already present under `SKETCH_FIX_DIR`; streams the optimization graph and writes tree-search traces under `TRACE_DIR`.

## Benchmarks and workloads

- **HITS-style**: Queries come from `benchmark/hits_queries.sql` and are loaded by `benchmark/click_benchmark.py`.
- **SSB**: See `benchmark/ssb_benchmark.py`.

When switching benchmarks, update `BENCHMARK`, `DB_JSON`, and related paths in `config.py` together.

## Notes

- Paths in `config.py` (local DuckDB files, Docker volume mounts) are machine-specific; re-check them when moving to another host.
- Some modules (e.g. `dbweaver/sketch/plan.py`) assume imports relative to runtime `PYTHONPATH`; if you see `ModuleNotFoundError`, run from the project root or adjust `sys.path`.
- Generation and optimization call external APIs and may keep Docker / CPU busy for a long time; they are best suited to experimental environments.

## License

If you publish this project publicly, add a `LICENSE` file and update this section accordingly.
