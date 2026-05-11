# 🪟 Running MMORE on Windows

## Overview

MMORE was developed and tested mainly on Linux. It runs on Windows too, but a few things behave differently. This page lists those differences and the fix for each one.

If you work on Linux or macOS, you can skip this page.

## 1. `milvus-lite` is not available on Windows

The example configs use a local `.db` file (`uri: ./proc_demo.db`), which relies on `milvus-lite`. There is no Windows build of `milvus-lite`, so you will see:

```
ModuleNotFoundError: No module named 'milvus_lite'
```

### Fix: run Milvus in Docker

Use the `docker-compose` file to start Milvus together with its dependencies (`etcd` and `minio`):

```powershell
docker compose -f docker-compose-milvus.yml up -d
```

Wait about a minute (the health check has `start_period: 90s`), then confirm the containers are up:

```powershell
docker ps
```

You should see three containers with `(healthy)` status.

### Create the database

MMORE does not create the database automatically when connecting to a remote Milvus. Run this once:

```powershell
python -c "from pymilvus import connections, db; connections.connect(uri='http://127.0.0.1:19530'); db.create_database('my_db')"
```

### Point the configs at the Docker instance

In both `examples/index/config.yaml` and `examples/retriever_api/config.yaml`, change:

```yaml
db:
  uri: ./proc_demo.db
  name: my_db
```

to:

```yaml
db:
  uri: http://127.0.0.1:19530
  name: my_db
```

## 2. Surya OCR can crash the process on large PDFs

When processing large PDFs, the surya-based OCR may crash with:

```
Process finished with exit code 0xC0000005
```

This is a hard crash inside a native dependency. On Windows, use the fast processors instead, which rely on PyMuPDF rather than surya.

In your `process` config:

```yaml
use_fast_processors: true
```

You lose some accuracy on heavily scanned PDFs, but the pipeline no longer crashes.

## 3. Keep the collection name consistent

`examples/index/config.yaml` and `examples/retriever_api/config.yaml` each have a `collection_name` field. If they differ, the retriever starts fine but later fails with:

```
ValueError: The Milvus database has not been initialized yet / does not have a collection ...
```

Use the same value (for example `my_docs`) in both files.

## 4. Check that the setup works

Once Milvus is running and the configs use the same collection name, confirm the connection:

```powershell
python -c "from pymilvus import MilvusClient; c = MilvusClient(uri='http://127.0.0.1:19530', db_name='my_db'); print(c.list_collections())"
```

This returns a list of collections (empty before you index anything).

## See also

- [Installation](installation.md)
- [Quickstart](quickstart.md)
