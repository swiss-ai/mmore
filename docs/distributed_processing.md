# Distributed Document Processing Guide

This guide explains how to set up and run distributed document processing for the RAG system across multiple nodes.

## Overview

Distributed processing allows you to scale document indexing across multiple machines, significantly improving processing speed for large document collections. The system uses Dask for distributed task scheduling and execution.

## Prerequisites

- Multiple machines/nodes with network connectivity
- Python environment on each node
- Access to a shared filesystem or the ability to copy files between nodes

## Setup Process

### 1. Prepare Your Configuration File

Check your processing configuration file ([example](/examples/process/config.yaml)), to include the distributed settings:

```yaml
dispatcher_config:
  distributed: true
  scheduler_file: "/path/to/scheduler.json"  # Shared location accessible by all nodes
```

Other important configuration options:
- `input_folder`: Path to your documents
- `output_folder`: Where processed results will be stored
- `use_fast_processors`: Set to `true` for faster processing (may reduce accuracy)

### 2. Install Dependencies on all Nodes

On each node, run:

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd mmore

# Make a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 3. Launch the Distributed Processing

#### Step 1: Start the Master Node (Rank 0)

```bash
bash scripts/process_distributed.sh --config-file /path/to/config.yaml --rank 0
```

The master node will:
- Start the Dask scheduler
- Launch a worker process
- Prompt you to start the processing when ready

#### Step 2: Start Worker Nodes (Rank > 0)

On each additional node, run:

```bash
bash scripts/process_distributed.sh --config-path /path/to/config.yaml --rank 1
```

Replace `rank 1` with a unique rank number for each node (1, 2, 3, etc.). The node should be ready in a matter of 5 seconds.

#### Step 3: Begin Processing

Once all nodes are running, return to the master node and type `go`. The master node proceeds to crawl the input folder, split the workload among connected nodes and make them start their work.

The dask server will be automatically shut down by the master node at the end of the processing. This will also shut down the dask workers on all the connected nodes.

## Monitoring Progress

You can monitor the processing using the dashboard, just check its [documentation](./dashboard.md).

The dashboard provides:
- Real-time progress visualization
- Worker status monitoring
- The ability to gracefully stop workers

## Output Structure

After processing completes, the output will be organized as follows:

```
output_folder/
├── processors/
│   ├── Processor_type_1/
│   │   └── results.jsonl
│   ├── Processor_type_2/
│   │   └── results.jsonl
│   └── ...
├── merged/
│   └── merged_results.jsonl
└── images/
```

## Troubleshooting

- **Workers not connecting**: Ensure all nodes can access the scheduler file location
- **Processing errors**: Check logs on the master node
- **Performance issues**: Adjust batch sizes and worker counts in the configuration

## Advanced Configuration

For optimal performance, consider adjusting:
- Processor batch sizes
- Number of threads per worker
- Memory limits for workers

Refer to the [process documentation](./process.md) for more details on configuration options.
