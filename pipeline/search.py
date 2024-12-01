import os
from transformers import BertTokenizer, BertModel
import sqlite3
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set the environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to convert text to BERT embeddings
def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Load vectors from the database
conn = sqlite3.connect('vectors.db')
c = conn.cursor()
c.execute("SELECT id, vector FROM vectors")
rows = c.fetchall()

# Convert vectors back to numpy arrays
ids = []
vectors = []
for row in rows:
    ids.append(row[0])
    vectors.append(np.frombuffer(row[1], dtype=np.float32))

vectors = np.vstack(vectors)

# Build faiss index
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Search function
def search(query, top_k=5):
    query_vector = text_to_vector(query)
    D, I = index.search(query_vector, top_k)
    result_ids = [ids[i] for i in I[0]]
    
    # Retrieve original text data
    placeholders = ','.join('?' for _ in result_ids)
    c.execute(f"SELECT id, reviewText FROM reviews WHERE id IN ({placeholders})", result_ids)
    results = c.fetchall()
    
    # Calculate cosine similarity scores
    result_vectors = np.vstack([vectors[ids.index(id)] for id in result_ids])
    similarities = cosine_similarity(query_vector, result_vectors).flatten()
    
    # Combine results with similarity scores
    results_with_scores = [(result, similarities[i]) for i, result in enumerate(results)]
    
    return results_with_scores

# Example search
if __name__ == "__main__":
    query = "interesting activity on land and under the water"
    results = search(query)
    print("Search results for query:", query)
    for result, score in results:
        print(f"Result: {result}, Similarity Score: {score}")