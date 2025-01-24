import os
from transformers import BertTokenizer, BertModel
import sqlite3
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set the environment variable to avoid OpenMP runtime errors on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to convert input text into BERT embeddings
def text_to_vector(text):
    # Tokenize and prepare the text for the BERT model
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    # Get the output embeddings from BERT
    outputs = model(**inputs)
    # Return the mean of the last hidden state as a single embedding
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Load vectors and IDs from the SQLite database
conn = sqlite3.connect('vectors.db')
c = conn.cursor()
c.execute("SELECT id, vector FROM vectors")
rows = c.fetchall()

# Convert the retrieved database vectors from binary blobs to numpy arrays
ids = []  # List to store the review IDs
vectors = []  # List to store the corresponding vectors
for row in rows:
    ids.append(row[0])  # Append the review ID
    vectors.append(np.frombuffer(row[1], dtype=np.float32))  # Convert the binary blob to numpy array

# Stack all vectors into a 2D numpy array
vectors = np.vstack(vectors)

# Create and populate a FAISS index for efficient vector similarity search
index = faiss.IndexFlatL2(vectors.shape[1])  # L2 distance metric
index.add(vectors)  # Add the vectors to the index

# Function to perform a search query
def search(query, top_k=5):
    # Convert the query text into a vector
    query_vector = text_to_vector(query)
    # Search the FAISS index for the top_k nearest neighbors
    D, I = index.search(query_vector, top_k)  # D: distances, I: indices of neighbors
    # Retrieve the IDs of the top_k results
    result_ids = [ids[i] for i in I[0]]
    
    # Retrieve the original text data corresponding to the result IDs
    placeholders = ','.join('?' for _ in result_ids)  # Create a dynamic query placeholder string
    c.execute(f"SELECT id, reviewText FROM reviews WHERE id IN ({placeholders})", result_ids)
    results = c.fetchall()
    
    # Calculate cosine similarity between the query vector and the result vectors
    result_vectors = np.vstack([vectors[ids.index(id)] for id in result_ids])  # Fetch result vectors
    similarities = cosine_similarity(query_vector, result_vectors).flatten()
    
    # Combine the results with their similarity scores
    results_with_scores = [(result, similarities[i]) for i, result in enumerate(results)]
    
    return results_with_scores

# Example search usage
if __name__ == "__main__":
    # Define a query string
    query = "interesting activity on land and under the water"
    # Perform the search
    results = search(query)
    # Display the results
    print("Search results for query:", query)
    for result, score in results:
        print(f"Result: {result}, Similarity Score: {score}")
