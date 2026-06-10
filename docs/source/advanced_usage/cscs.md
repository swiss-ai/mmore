# 🇨🇭 CSCS Alps quickstart

This page walks through using mmore on CSCS Alps (Clariden, GH200 aarch64,
64 KB page size) with the **Qdrant backend** introduced by
[PR #283](https://github.com/swiss-ai/mmore/pull/283). Two paths are
covered:

- **Embedded mode** — Qdrant runs inside the Python process. No external
  server, no Rust binary, no jemalloc. Recommended for development,
  smoke tests, and small/medium collections.
- **Server mode** — a real Qdrant Rust binary, started alongside your
  job. Needed for collections above ~10k ColPali pages or ~50k text
  chunks. Requires compiling Qdrant from source on Alps (commands below).

## Why aarch64 / 64 KB pages matter

```{important}
On Alps GH200 nodes, the upstream **prebuilt Qdrant server binary**
crashes immediately with
`<jemalloc>: Unsupported system page size`. jemalloc is statically
linked and was compiled assuming 4 KB pages; GH200 uses 64 KB pages.
**Embedded mode bypasses this** because it never loads the Rust binary
(it is a pure-Python reimplementation shipped with `qdrant-client`).
For server mode you have to rebuild Qdrant once, with
`JEMALLOC_SYS_WITH_LG_PAGE=16`. The full recipe is in
[§ Server mode](#server-mode-optional-for-large-collections) below.
```

---

## Embedded mode (recommended default)

### 1. Allocate a node

```bash
salloc --account=<your-account> --partition=normal --gpus=1 --time=01:00:00
```

### 2. Set up the Python environment

```bash
# Put caches on scratch — home quota is small on Alps.
export HF_HOME=/iopsstor/scratch/cscs/$USER/hf_cache
export PIP_CACHE_DIR=/iopsstor/scratch/cscs/$USER/pip_cache

# venv
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[qdrant]"
```

### 3. Configure the indexer

Set `indexer.db.backend: qdrant` and point `indexer.db.uri` at a writable
scratch path — that directory is where the embedded Qdrant stores its data.

```yaml
# examples/index/config.yaml
indexer:
  dense_model:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    is_multimodal: false
  sparse_model:
    model_name: splade
    is_multimodal: false
  db:
    backend: qdrant
    uri: /iopsstor/scratch/cscs/${USER}/qdrant_data
    name: my_db
collection_name: my_docs
documents_path: examples/postprocessor/outputs/merged/results.jsonl
```

### 4. Run the indexer

```bash
python3 -m mmore index --config-file examples/index/config.yaml
```

First run creates the collection under `indexer.db.uri`; subsequent runs
append.

### 5. Query

Mirror the same `db.backend: qdrant` block in your RAG config (under
`rag.retriever.db`) and run:

```bash
python3 -m mmore rag --config-file examples/rag/config.yaml
```

### 6. Inspect what landed

```bash
ls /iopsstor/scratch/cscs/$USER/qdrant_data/
# → collection/<collection-name>/storage/ — Qdrant's on-disk segments
```

### When to outgrow embedded mode

Embedded Qdrant is pure-Python and slow at scale. Rule of thumb:

| corpus size | embedded mode? |
|---|---|
| < 5k text chunks or < 2k ColPali pages | yes — keep it simple |
| 5k–50k chunks / 2k–10k ColPali pages | usable but noticeably slow |
| > 50k chunks or > 10k ColPali pages | switch to server mode (below) |

---

## Server mode (optional, for large collections)

Server mode runs the real Qdrant Rust binary alongside your job, bound
to `127.0.0.1` so it never leaves the SLURM allocation. The binary
itself has to be compiled once on an Alps node (the prebuilt one
crashes — see the box above). After that, every job just launches it
and points the client at `http://127.0.0.1:6333`.

### 1. Build the Qdrant binary (one-time, ~20 min)

These commands assume you are on an Alps login or compute node with
network access. Adjust `$QDRANT_PREFIX` to wherever you want the source
tree + binary to live.

```bash
export QDRANT_PREFIX=/iopsstor/scratch/cscs/$USER/qdrant
mkdir -p "$QDRANT_PREFIX" && cd "$QDRANT_PREFIX"

# --- Install Rust toolchain (skip if cargo is already on PATH) ---
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path
. "$HOME/.cargo/env"

# --- Install protoc (build dep, no aarch64 package on most distros) ---
PROTOC_VERSION=34.1
curl -fsSL -o protoc.zip \
    "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-aarch_64.zip"
unzip -oq protoc.zip -d protoc && rm protoc.zip
export PROTOC="$QDRANT_PREFIX/protoc/bin/protoc"

# --- Clone Qdrant source at a pinned tag ---
QDRANT_VERSION=v1.17.1
git clone --depth 1 --branch "$QDRANT_VERSION" https://github.com/qdrant/qdrant.git src
cd src

# --- Build with the 64KB-page jemalloc fix ---
# log2(65536) = 16. This is the only non-default flag; the rest is upstream Qdrant.
JEMALLOC_SYS_WITH_LG_PAGE=16 cargo build --release --bin qdrant -j 64

# --- Verify ---
./target/release/qdrant --version
# → qdrant 1.17.1
```

The resulting binary is `$QDRANT_PREFIX/src/target/release/qdrant` (~72 MB).
It is **Alps-specific** — it works on any GH200 node but will crash on
4 KB-page systems (laptops, most cloud VMs). To upgrade Qdrant later,
re-run the last two blocks with a different `QDRANT_VERSION`.

### 2. Launch the binary alongside your workload

Inside any SLURM job, start the server in the background bound to
loopback, wait for `/healthz`, then run the rest of your pipeline.
A `trap` ensures it is killed cleanly on job exit.

```bash
#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --gpus=1
#SBATCH --time=01:00:00

set -euo pipefail

QDRANT_BIN=/iopsstor/scratch/cscs/$USER/qdrant/src/target/release/qdrant
QDRANT_DATA=/iopsstor/scratch/cscs/$USER/qdrant_server_data
mkdir -p "$QDRANT_DATA"

# Clean up the server on job exit (success or failure)
trap '[ -n "${QDRANT_PID:-}" ] && kill "$QDRANT_PID" 2>/dev/null || true' EXIT

QDRANT__SERVICE__HOST=127.0.0.1 \
QDRANT__SERVICE__MAX_WORKERS=4 \
QDRANT__STORAGE__STORAGE_PATH="$QDRANT_DATA" \
    "$QDRANT_BIN" &> qdrant.log &
QDRANT_PID=$!

# Wait for the server to come up (~1–2 s typically).
for i in $(seq 1 15); do
    sleep 1
    if curl -fsS http://127.0.0.1:6333/healthz > /dev/null 2>&1; then
        echo "qdrant ready after ${i}s"
        break
    fi
done

# --- Your mmore commands go here. Use the same config as embedded mode
#     but point db.uri at the running server: ---
#       indexer.db.uri: http://127.0.0.1:6333
python3 -m mmore index --config-file examples/index/config.yaml
python3 -m mmore rag --config-file examples/rag/config.yaml
```

### 3. Server-mode config

Identical to embedded mode except `indexer.db.uri` is an HTTP URL:

```yaml
indexer:
  dense_model:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    is_multimodal: false
  sparse_model:
    model_name: splade
    is_multimodal: false
  db:
    backend: qdrant
    uri: http://127.0.0.1:6333    # ← server mode
    name: my_db
collection_name: my_docs
documents_path: examples/postprocessor/outputs/merged/results.jsonl
```

The `QdrantMilvusClient` adapter detects the `http://` prefix
automatically and switches to server mode; no other code changes are
needed.

---

## Troubleshooting

- **`<jemalloc>: Unsupported system page size` at server startup** — you
  ran the upstream prebuilt binary instead of the one you compiled.
  Check `which qdrant` / the absolute path in your sbatch.
- **`Address already in use` on port 6333** — another `qdrant` process
  on the same node is using it. Either kill it (`pkill qdrant`) or set
  `QDRANT__SERVICE__HTTP_PORT=<free port>`.
- **Embedded-mode queries take seconds at >10k vectors** — that is
  expected; switch to server mode.
- **Home directory quota errors during `uv pip install`** — make sure
  `PIP_CACHE_DIR` and `HF_HOME` point at `/iopsstor/scratch/...`.
