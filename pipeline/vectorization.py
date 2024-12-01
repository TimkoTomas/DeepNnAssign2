import pandas as pd
from transformers import BertTokenizer, BertModel
import sqlite3
import faiss
import numpy as np
from tqdm import tqdm

df = pd.read_csv('movie1000.csv')

# Preprocess the 'reviewText' column (e.g., remove NaNs, lowercasing, etc.)
df['reviewText'] = df['reviewText'].fillna('').astype(str).str.lower()

# Ensure 'reviewId' is unique by adding a unique suffix if necessary
df['reviewId'] = df['reviewId'].astype(str) + '_' + df.groupby('reviewId').cumcount().astype(str)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

"""
# Load fine-tuned BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./fine-tuned-bert')
model = BertModel.from_pretrained('./fine-tuned-bert')
"""

# Function to convert text to BERT embeddings
def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Convert reviewText to vectors with progress bar
vectors = []
for text in tqdm(df['reviewText'], desc="Creating vectors"):
    vectors.append(text_to_vector(text))
df['vector'] = vectors

# Create SQLite database
conn = sqlite3.connect('vectors.db')
c = conn.cursor()

# Create table for original data
c.execute('''CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, reviewText TEXT)''')

# Create table for vectors
c.execute('''CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, vector BLOB)''')

# Insert original data into the database with progress bar
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Inserting reviews"):
    c.execute("INSERT INTO reviews (id, reviewText) VALUES (?, ?)", (row['reviewId'], row['reviewText']))

# Insert vectors into the database with progress bar
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Inserting vectors"):
    vector = row['vector'].tobytes()
    c.execute("INSERT INTO vectors (id, vector) VALUES (?, ?)", (row['reviewId'], vector))

conn.commit()
conn.close()