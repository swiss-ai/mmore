# 🇨🇭 CSCS Alps quickstart

This page walks through using mmore on CSCS Alps (Clariden, GH200 aarch64,
64 KB page size) with the **Qdrant backend in embedded mode** introduced by
[PR #283](https://github.com/swiss-ai/mmore/pull/283). Embedded mode runs
inside the Python process with no external server — ideal for development,
smoke tests, and small/medium collections.

```{important}
The prebuilt Qdrant **server** binary for aarch64 ships a jemalloc compiled
for 4 KB pages and crashes on GH200 with
`<jemalloc>: Unsupported system page size`. Embedded mode bypasses this
because it never loads the Rust binary. For server-mode workloads on Alps
you need a custom Qdrant build (see
[qdrant-alps](https://github.com/jeremydoumeng/qdrant-alps)).
```

## 1. Allocate a node

```bash
salloc --account=<your-account> --partition=normal --gpus=1 --time=01:00:00
```

## 2. Set up the Python environment

```bash
# put caches on scratch (home quota is small)
export HF_HOME=/iopsstor/scratch/cscs/$USER/hf_cache
export PIP_CACHE_DIR=/iopsstor/scratch/cscs/$USER/pip_cache

# venv
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[qdrant]"
```

## 3. Configure the indexer

Set `db.backend: qdrant` and point `db.uri` at a writable scratch path —
that directory is where the embedded Qdrant stores its data.

```yaml
# examples/index/config.yaml
documents_path: /iopsstor/scratch/cscs/${USER}/docs
dense_model:
  model_name: sentence-transformers/all-MiniLM-L6-v2
sparse_model:
  model_name: splade
db:
  backend: qdrant
  uri: /iopsstor/scratch/cscs/${USER}/qdrant_data
  name: my_db
collection_name: my_docs
```

## 4. Run the indexer

```bash
python -m mmore.run_index --config examples/index/config.yaml
```

First run creates the collection under `db.uri`; subsequent runs append.

## 5. Query

Mirror the same `db.backend: qdrant` block in the RAG config and run:

```bash
python -m mmore.run_rag --config examples/rag/config.yaml
```

## 6. Inspect what landed

```bash
ls /iopsstor/scratch/cscs/$USER/qdrant_data/
# → collection/<collection-name>/storage/ — Qdrant's on-disk segments
```

## When to outgrow embedded mode

Embedded Qdrant is pure-Python and slow at scale. Rule of thumb:

| corpus size | embedded mode? |
|---|---|
| < 5k text chunks or < 2k ColPali pages | yes — keep it simple |
| 5k–50k chunks / 2k–10k ColPali pages | usable but noticeably slow |
| > 50k chunks or > 10k ColPali pages | switch to server mode (custom Alps build) |

For the server-mode path on Alps, see
[qdrant-alps](https://github.com/jeremydoumeng/qdrant-alps).
