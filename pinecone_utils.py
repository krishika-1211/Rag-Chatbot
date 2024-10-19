import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import List

load_dotenv()
# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")  # Ensure the API key is loaded from the environment
)

# Define your index name and dimension
index_name = "myindex"
dimension = 384

pc.delete_index(index_name)

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f"Index '{index_name}' created")
else:
    print(f"Index '{index_name}' already exists.")

def store_vectors(embeddings, prompt):
    try:
        index = pc.Index(index_name)
        to_upsert = [{"id": prompt, "values": embeddings}]
        response = index.upsert(vectors=to_upsert)
        print(f"Vectors upserted successfully: {response}")
    except Exception as e:
        print(f"Error while storing vectors: {e}")


# Function to query Pinecone for similar vectors
def query_pinecone(query_vector, top_k=5):
    try:
        index = pc.Index(index_name)  # Connect to the index
        # results = index.query(queries=[query_vector], top_k=top_k)
        results = index.query(vector=query_vector, top_k=top_k, include_values=True)
        print("Query executed successfully.")
        # if 'matches' not in results or results['matches'] is None:
        #     return []
        print(f"Results: {results}")
        return results['matches']
    except Exception as e:
        print(f"Error during Pinecone query: {e}")
