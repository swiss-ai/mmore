# --- Set up and Run the RAG Pipeline ---
from mmore.rag.pipeline import RAGPipeline, RAGConfig
from mmore.rag.llm import LLMConfig
from mmore.rag.retriever import RetrieverConfig
from mmore.index.indexer import DBConfig

from mmore.rag.model import DenseModel, SparseModel
import mmore.rag.retriever as retriever_mod

# Patch the retriever module's namespace.
retriever_mod.__dict__['DenseModel'] = DenseModel
retriever_mod.__dict__['SparseModel'] = SparseModel




rag_config = RAGConfig(
    retriever=RetrieverConfig(
         db=DBConfig(uri="demo.db", name="my_db"),
         hybrid_search_weight=0.5,
         k=1
    ),
    llm=LLMConfig(
        llm_name="microsoft/DialoGPT-medium",  # Chat-enabled model
        max_new_tokens=50,
        temperature=0.7
    ),
    system_prompt="""\
Use the following context to answer the question.
Context:
{context}
Question:
{input}
Answer:"""
)


# Create the RAG pipeline from the configuration.
rag = RAGPipeline.from_config(rag_config)

# Define a sample query that should retrieve the indexed document.
query = "How do I sync my calendar?"

# The RAG pipeline expects a dictionary with 'input', 'collection_name', and (optionally) 'partition_name'.
rag_output = rag({"input": query, "collection_name": "my_docs", "partition_name": None}, return_dict=True)

print("RAG output:")
print(rag_output)