# :robot: MMORE RAG 

## :bulb: TL;DR

> The `RAG` module enables the creation of a modular RAG inference pipeline for your indexed multimodal documents, using two inference modes:
> 1. **API**: Creates a server hosting the pipeline
> 2. **Local**: Runs the inference locally (:warning: might be long when running local models :warning:) 
> 
> You can customize various parts of the pipeline by defining [an inference RAG configuration file](/examples/rag/api/rag_api.yaml).

## :computer: Minimal Example:

Here is a minimal example to create a RAG pipeline hosted through [LangServe](https://python.langchain.com/docs/langserve/) servers.

1. Create your RAG Inference config file based on the [local example](/examples/rag/config.yaml) or the [API example](/examples/rag/config_api.yaml). You can check the structure of the configuration file with the dataclass [RAGConfig](/src/mmore/rag/pipeline.py).

2. Start your RAG pipeline using the `run_rag.py` script and your config file
    ```bash
    python3 -m mmore rag --config_file /path/to/config.yaml
    ```

3. In API mode, query the server like any other LangServe server:
    ```bash
    curl --location --request POST http://localhost:8000/rag/invoke \
    -H 'Content-Type: application/json' \
    -d '{
        "input": {
            "input": "What is Meditron?",
            "collection_name": "my_docs"
        }
    }'
    ```

    ```bash
    curl --location --request GET http://localhost:8000/rag/input_schema \
    -H 'Content-Type: application/json' 
    ```

    In local mode, the pipeline is run directly with the input data specified in the configuration file and the result is saved at the specified path.

See [`examples/rag`](/examples/rag/) for other use cases.

## :mag: Modules

The RAG decomposes into two main modules:
1. The `Retriever`, which retrieves multimodal documents from the database. 
2. The `LLM`, which wraps different types of multimodal-able LLMs.

#### Retriever

Here is an example on how to use the retriever module alone. Note that it assumes that you already created a DB using [the indexer module](index.md).

1. Create a config based on the [example config file](/examples/index/config.yaml)

2. Retrieve on the vector store using the `Retriever` class:
    ```python
    from mmore.rag.retriever import Retriever

    # Create the Retriever
    retriever = Retriever.from_config('/path/to/your/retriever_config.yaml')

    # Retrieves the top 3 documents using an hybrid approach (e.g. dense + sparse embeddings)
    retriever.retrieve(
        'What is Meditron?',
        k=3,
        collection_name="my_docs",
        search_type="hybrid"  # Options: "dense", "sparse", "hybrid"
    )
    ```

#### LLM

Here is an example on how to use the `LLM` module alone. Note that it assumes that you already created a DB using [the indexer module](index.md).

1. Create a config file:
    ```yaml
    llm_name: gpt-4o-mini
    max_new_tokens: 150
    temperature: 0.7
    ```

2. Query the LLM:
    ```python
    from mmore.rag.llm import LLM

    # Create the LLM
    llm = LLM.from_config('/path/to/your/llm_config.yaml')

    # Create your messages
    messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    (
        "human",
        "I love Meditron."
    ),
    ]

    # Retrieves the top 3 documents using an hybrid approach (e.g. dense + sparse embeddings)
    llm.invoke(messages)
    ```
## :wrench: Customization

Our RAG pipeline is built to take full advantage of [LangChain](https://python.langchain.com/docs/introduction/) abstractions, providing compatibility with all components offered.

#### Retriever

Our retriever is a LangChain [`BaseRetriever`](https://python.langchain.com/api_reference/core/retrievers/langchain_core.retrievers.BaseRetriever.html). If you want to create a custom retriever (e.g. GraphRetriever,...) you can simply make it inherit from this class and use it as described in our examples.

#### LLM

Our LLMs are LangChain's [`BaseChatModel`](https://python.langchain.com/api_reference/core/retrievers/langchain_core.retrievers.BaseRetriever.html) base class. If you want to create a custom retriever you can simply make it inherit from this class and use it as described in our examples. 

> :warning: Note that we support [HuggingFace Hub](https://huggingface.co/models) models, so a simpler solution is to push a model to the hub and use the class as defined.
