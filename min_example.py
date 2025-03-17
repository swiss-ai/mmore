from mmore.process.processors.pdf_processor import PDFProcessor 
from mmore.process.processors.base import ProcessorConfig
from mmore.type import MultimodalSample

# --- Additional imports for the indexer ---
from mmore.index.indexer import Indexer, IndexerConfig
from mmore.rag.model import DenseModelConfig, SparseModelConfig

pdf_file_paths = ["examples/sample_data/pdf/calendar.pdf"]
out_file = "results/example.jsonl"

pdf_processor_config = ProcessorConfig(custom_config={"output_path": "results"})
pdf_processor = PDFProcessor(config=pdf_processor_config)
result_pdf = pdf_processor.process_batch(pdf_file_paths, True, 1) # args: file_paths, fast mode (True/False), num_workers

MultimodalSample.to_jsonl(out_file, result_pdf)

# --- Set up the indexer configuration ---
# I use Salesforce/blip2-flan-t5-xl which is a model that captures information from text and images
# I use splade sparse embedding model 
dense_model_config = DenseModelConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", is_multimodal=False)
sparse_model_config = SparseModelConfig(model_name="splade", is_multimodal=True)

# Create an indexer configuration including database settings
# Note that by default, the example code sets db={"uri": "demo.db", "name": "my_db"} in the IndexerConfig. That implies it’s using a LOCAL Milvus setup (local file-based mode) -> db is stored locally
indexer_config = IndexerConfig(
    dense_model=dense_model_config,
    sparse_model=sparse_model_config,
    db={"uri": "demo.db", "name": "my_db"}
)

# --- Index the processed documents ---
# This convenience method will:
# 1. Instantiate the Indexer from the provided configuration.
# 2. Create a Milvus collection (if it does not exist).
# 3. Compute embeddings (both dense and sparse) for each document.
# 4. Insert the documents in batches into the Milvus vector database.
indexer = Indexer.from_documents(
    config=indexer_config, 
    documents=result_pdf, 
    collection_name='my_docs',
    partition_name=None,  # Optional: specify a partition if needed
    batch_size=64         # You can adjust the batch size based on your data
)

print(f"Indexed {len(result_pdf)} documents into Milvus.")