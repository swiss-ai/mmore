# MMORE Documentation

MMORE is an open-source multimodal ingestion and retrieval framework designed for heterogeneous document collections.

It provides tools to process documents, build indexes, run retrieval pipelines, support multimodal workflows, and run distributed processing for larger collections and production-oriented settings.

## What is MMORE?

MMORE helps you build retrieval systems over complex document collections by combining:

- document ingestion and processing
- indexing pipelines
- retrieval and RAG workflows
- multimodal retrieval support
- distributed processing support for large-scale document ingestion
- evaluation and profiling tools


## Where to start

Depending on what you want to do, start in different places:

- to install MMORE, read [Installation](getting_started/installation.md)
- to run a first workflow, read [Quickstart](getting_started/quickstart.md)
- to understand the overall system, read [Architecture](getting_started/architecture.md)
- to understand ingestion and indexing, read [Process](getting_started/process.md) and [Indexing](getting_started/indexing.md)
- to work on retrieval workflows, read [RAG](getting_started/rag.md)
- to work on multimodal retrieval, read [ColPali](core_features/colpali.md)
- to run distributed processing, read [Distributed processing](advanced_usage/distributed_processing.md)
- to contribute to the codebase, read [For developers](developer_documentation/for_devs.md)

## Documentation map

```{toctree}
:maxdepth: 1
:caption: Getting started

getting_started/installation
getting_started/quickstart
getting_started/architecture
getting_started/process
getting_started/indexing
getting_started/rag
```

```{toctree}
:maxdepth: 1
:caption: Core features

core_features/colpali
core_features/websearch
core_features/evaluation
```

```{toctree}
:maxdepth: 1
:caption: Advanced usage

advanced_usage/uv
advanced_usage/distributed_processing
advanced_usage/profiler
advanced_usage/rcp_and_production
```

```{toctree}
:maxdepth: 1
:caption: Developer documentation

developer_documentation/for_devs
developer_documentation/index_api
```

## Page guide

Here is a quick overview of the main pages:

- [Installation](getting_started/installation.md): set up MMORE and prepare your environment
- [Quickstart](getting_started/quickstart.md): run a first minimal workflow end to end
- [Architecture](getting_started/architecture.md): understand the main system components and how they interact
- [Processing pipeline](getting_started/process.md): understand how documents are ingested and transformed
- [Indexing](getting_started/indexing.md): build and manage indexes
- [RAG](getting_started/rag.md): structure retrieval-augmented generation workflows
- [ColPali](core_features/colpali.md): multimodal retrieval-related documentation
- [Websearch](core_features/websearch.md): web search integration and related workflows
- [Evaluation](core_features/evaluation.md): assess system performance
- [Distributed processing](advanced_usage/distributed_processing.md): scale processing across larger workloads
- [Profiler](advanced_usage/profiler.md): profile and analyze performance
- [uv](advanced_usage/uv.md): environment and dependency workflow
- [Cluster and production](advanced_usage/rcp_and_production.md): deployment and production-oriented guidance
- [For developers](developer_documentation/for_devs.md): contributor and internal development documentation
- [Index API](developer_documentation/index_api.md): API-oriented reference for indexing-related functionality

