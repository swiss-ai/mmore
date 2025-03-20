# :robot: MMORE Index 
## :bulb: TL;DR
> The `Index` module handles the indexing and post-processing of the extracted data from the multimodal documents. It creates an indexed Vector Store DB based on [Milvus](https://milvus.io/). We enable the use of *hybrid* retrieval, combining both *dense* and *sparse* retrieval.
>
> You can customize various parts of the pipeline by defining [an inference indexing config file](../examples/index/config.yaml).

## :computer: Minimal Example:
Here is a minimal example to index [processed documents](process.md).
1. Create a config file:
    ```yaml
    indexer:
        dense_model_name: sentence-transformers/all-MiniLM-L6-v2
        sparse_model_name: splade
        db:
            uri: ./proc_demo.db
            name: my_db
    collection_name: my_docs
    documents_path: './output'
    ```

2. Index your documents by calling the inference script:
    ```bash
    python run_index.py --config_file /path/to/config.yaml
    ```
See [`examples/index`](../examples/index/) for other examples.