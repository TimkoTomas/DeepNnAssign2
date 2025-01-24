# DeepNnAssign2

# Sentiment Search Pipeline

## Project Description
The Sentiment Search Pipeline is a robust NLP project that combines semantic search and sentiment analysis. It uses pre-trained BERT models for generating text embeddings and classifying sentiment. The pipeline retrieves movie reviews that are semantically similar to a user-provided query and classifies the sentiment of each review as "Good" or "Bad" with a confidence score.

### Features
- **Semantic Search**: Uses FAISS for efficient retrieval of semantically similar reviews.
- **Sentiment Analysis**: Analyzes and classifies the sentiment of retrieved reviews.
- **Interactive Visualization**: Plots similarity scores vs. sentiment confidence for different queries.

## Folder Structure
```plaintext
pipeline/
├── vectorization.py       # Preprocesses and vectorizes movie reviews
├── search.py              # Performs semantic search on vectorized data
├── search_sentiment.py    # Combines semantic search with sentiment analysis
├── vectors.db             # SQLite database with vectorized reviews
├── requirements.txt       # Python dependencies
