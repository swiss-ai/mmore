# --- Querying Milvus local DB

# --- Additional imports for querying created Milvus Collection
from pymilvus import MilvusClient

# Connect to your local Milvus instance
client = MilvusClient(uri="demo.db", db_name="my_db")

collection_name = "my_docs"

# Get collection statistics
stats = client.get_collection_stats(collection_name)
print("Collection Stats:", stats)

# Query the collection for all documents
results = client.query(
    collection_name,  # positional argument for the collection name
    "",               # positional argument for the query expression (empty to get all)
    output_fields=["id", "text", "dense_embedding", "sparse_embedding"],
    limit=10          # specify a limit, e.g. 10 documents
)

print("Stored documents:")
for record in results:
    print(record)

