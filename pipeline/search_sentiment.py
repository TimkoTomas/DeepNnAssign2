import os
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import sqlite3
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Set the environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Path to the fine-tuned sentiment analysis model
model_path = '../models/fine-tuned-bert'
print("Directory contents:", os.listdir(model_path))

# Load pre-trained BERT model and tokenizer for embeddings
embedding_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
embedding_model = BertModel.from_pretrained('bert-base-uncased')

# Load fine-tuned BERT model and tokenizer for sentiment classification
sentiment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentiment_model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
sentiment_model.eval()  # Set the model to evaluation mode for inference

# Function to convert text to BERT embeddings
def text_to_vector(text):
    # Tokenize and encode the input text for BERT embeddings
    inputs = embedding_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to classify the sentiment of a given text
def classify_sentiment(text):
    # Tokenize and encode the input text for sentiment classification
    inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    sentiment = torch.argmax(probabilities, dim=1).item()
    
    # Flip the sentiment labels (e.g., 0 -> "Good", 1 -> "Bad")
    sentiment_label = "Good" if sentiment == 0 else "Bad"
    return sentiment_label, probabilities[0][sentiment].item()  # Return label and confidence score

# Load vectors and review IDs from the SQLite database
conn = sqlite3.connect('vectors.db')
c = conn.cursor()
c.execute("SELECT id, vector FROM vectors")
rows = c.fetchall()

# Convert database vectors from binary blobs to numpy arrays
ids = []
vectors = []
for row in rows:
    ids.append(row[0])  # Store the review IDs
    vectors.append(np.frombuffer(row[1], dtype=np.float32))  # Decode binary blobs into numpy arrays

vectors = np.vstack(vectors)  # Stack all vectors into a 2D numpy array

# Create and populate a FAISS index for efficient similarity searches
index = faiss.IndexFlatL2(vectors.shape[1])  # Use L2 distance for the index
index.add(vectors)  # Add vectors to the index

# Function to perform a search query with sentiment analysis
def search(query, top_k=5):
    query_vector = text_to_vector(query)  # Convert the query text to a vector
    D, I = index.search(query_vector, top_k)  # Perform FAISS search (D: distances, I: indices)
    result_ids = [ids[i] for i in I[0]]  # Retrieve the IDs of the top-k results
    
    # Fetch the original review texts from the database
    placeholders = ','.join('?' for _ in result_ids)  # Create query placeholders for SQLite
    c.execute(f"SELECT id, reviewText FROM reviews WHERE id IN ({placeholders})", result_ids)
    results = c.fetchall()
    
    # Compute cosine similarity scores and analyze sentiment for each result
    results_with_scores_and_sentiment = []
    for i, (result, score) in enumerate(zip(results, cosine_similarity(query_vector, np.vstack([vectors[ids.index(id)] for id in result_ids])).flatten())):
        sentiment_label, confidence = classify_sentiment(result[1])  # Classify sentiment
        results_with_scores_and_sentiment.append({
            "id": result[0],
            "text": result[1],
            "similarity": score,
            "sentiment": sentiment_label,
            "confidence": confidence
        })
    
    return results_with_scores_and_sentiment

import matplotlib.pyplot as plt

if __name__ == "__main__":
    import pandas as pd

    # Example queries for different sentiments
    query_good = "a heartwarming story about friendship and hope"
    query_bad = "a poorly written plot with weak characters"
    query_neutral = "an average movie with some highs and lows"

    # Process each query and display results
    queries = {"Good": query_good, "Bad": query_bad, "Neutral": query_neutral}
    all_results = {}

    for sentiment, query in queries.items():
        print(f"\n--- Processing Query: {sentiment} ---\n")
        results = search(query, top_k=20)
        all_results[sentiment] = results
        for result in results:
            print(f"ID: {result['id']}, Text: {result['text']}")
            print(f"Similarity: {result['similarity']:.4f}, Sentiment: {result['sentiment']} "
                  f"(Confidence: {result['confidence']:.4f})\n")
        print("\n" + "-" * 50 + "\n")

    # Combine results for visualization
    data = []
    for sentiment, results in all_results.items():
        for result in results:
            data.append({
                "Query": sentiment,
                "ID": result["id"],
                "Text": result["text"],
                "Similarity": result["similarity"],
                "Sentiment": result["sentiment"],
                "Confidence": result["confidence"]
            })

    df = pd.DataFrame(data)

    # Visualization of similarity and sentiment confidence
    plt.figure(figsize=(10, 6))
    for sentiment in queries.keys():
        subset = df[df["Query"] == sentiment]
        plt.scatter(
            subset["Similarity"],
            subset["Confidence"],
            label=f"{sentiment} Query",
            s=100, alpha=0.7
        )
    
    plt.title("Sentiment Confidence vs Similarity")
    plt.xlabel("Similarity Score")
    plt.ylabel("Sentiment Confidence")
    plt.legend()
    plt.grid()
    plt.show()
