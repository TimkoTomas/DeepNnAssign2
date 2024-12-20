import os
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import sqlite3
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Set the environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os

model_path = '../models/fine-tuned-bert'
print("Directory contents:", os.listdir(model_path))

# Load pre-trained BERT model and tokenizer for embeddings
embedding_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
embedding_model = BertModel.from_pretrained('bert-base-uncased')

# Load pre-trained BERT model and tokenizer for sentiment classification
sentiment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentiment_model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)


sentiment_model.eval()  # Set the model to evaluation mode

# Function to convert text to BERT embeddings
def text_to_vector(text):
    inputs = embedding_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to classify sentiment of text with flipped labels
def classify_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    sentiment = torch.argmax(probabilities, dim=1).item()
    
    # Flip the sentiment labels
    sentiment_label = "Good" if sentiment == 0 else "Bad"  # Invert 0 -> Good, 1 -> Bad
    return sentiment_label, probabilities[0][sentiment].item()  # Label and confidence score


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

# Search function with sentiment analysis
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
    
    # Combine results with sentiment analysis and similarity scores
    results_with_scores_and_sentiment = []
    for i, (result, score) in enumerate(zip(results, similarities)):
        sentiment_label, confidence = classify_sentiment(result[1])  # Perform sentiment analysis
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
    import matplotlib.pyplot as plt
    import pandas as pd

    # Define individual queries
    query_good = "a heartwarming story about friendship and hope"
    query_bad = "a poorly written plot with weak characters"
    query_neutral = "an average movie with some highs and lows"

    # Process the first query (Good)
    print("\n--- Processing Query: Good ---\n")
    results_good = search(query_good, top_k=20)
    print(f"Query: {query_good}")
    for result in results_good:
        print(f"ID: {result['id']}, Text: {result['text']}")
        print(f"Similarity: {result['similarity']:.4f}, Sentiment: {result['sentiment']} "
              f"(Confidence: {result['confidence']:.4f})\n")
    print("\n" + "-" * 50 + "\n")

    # Process the second query (Bad)
    print("\n--- Processing Query: Bad ---\n")
    results_bad = search(query_bad, top_k=20)
    print(f"Query: {query_bad}")
    for result in results_bad:
        print(f"ID: {result['id']}, Text: {result['text']}")
        print(f"Similarity: {result['similarity']:.4f}, Sentiment: {result['sentiment']} "
              f"(Confidence: {result['confidence']:.4f})\n")
    print("\n" + "-" * 50 + "\n")

    # Process the third query (Neutral)
    print("\n--- Processing Query: Neutral ---\n")
    results_neutral = search(query_neutral, top_k=20)
    print(f"Query: {query_neutral}")
    for result in results_neutral:
        print(f"ID: {result['id']}, Text: {result['text']}")
        print(f"Similarity: {result['similarity']:.4f}, Sentiment: {result['sentiment']} "
              f"(Confidence: {result['confidence']:.4f})\n")
    print("\n" + "-" * 50 + "\n")

    # Combine all results for visualization
    all_results = {
        "Good": results_good,
        "Bad": results_bad,
        "Neutral": results_neutral
    }

    # Create a DataFrame for visualization
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

    # Visualization using matplotlib
    plt.figure(figsize=(10, 6))
    for sentiment in all_results.keys():
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
