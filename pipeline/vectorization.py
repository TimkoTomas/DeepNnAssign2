import pandas as pd
from transformers import BertTokenizer, BertModel
import sqlite3
import faiss
import numpy as np
from tqdm import tqdm

# Load the dataset of movie reviews
df = pd.read_csv('movie1000.csv')

# Preprocess the 'reviewText' column (e.g., replace NaNs, convert to lowercase)
df['reviewText'] = df['reviewText'].fillna('').astype(str).str.lower()

# Ensure 'reviewId' is unique by appending a suffix based on duplicate counts
df['reviewId'] = df['reviewId'].astype(str) + '_' + df.groupby('reviewId').cumcount().astype(str)

# Load the pre-trained BERT model and tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

"""
# Uncomment these lines if using a fine-tuned BERT model stored locally
tokenizer = BertTokenizer.from_pretrained('./models/fine-tuned-bert')
model = BertModel.from_pretrained('./models/fine-tuned-bert')
"""

# Function to convert text into BERT embeddings
def text_to_vector(text):
    # Tokenize the input text and prepare tensors for BERT
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    # Get the outputs from the BERT model
    outputs = model(**inputs)
    # Return the mean of the last hidden state (pooled representation)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Convert the 'reviewText' column to BERT embeddings with a progress bar
vectors = []
for text in tqdm(df['reviewText'], desc="Creating vectors"):
    vectors.append(text_to_vector(text))
df['vector'] = vectors  # Add the vectors as a new column in the DataFrame

# Create a SQLite database to store the data
conn = sqlite3.connect('vectors.db')
c = conn.cursor()

# Create a table to store the original reviews (id and text)
c.execute('''CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, reviewText TEXT)''')

# Create a table to store the vectors (id and vector blob)
c.execute('''CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, vector BLOB)''')

# Insert the original reviews into the database with a progress bar
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Inserting reviews"):
    c.execute("INSERT INTO reviews (id, reviewText) VALUES (?, ?)", (row['reviewId'], row['reviewText']))

# Insert the vector embeddings into the database with a progress bar
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Inserting vectors"):
    # Convert the numpy vector to a binary blob for storage
    vector = row['vector'].tobytes()
    c.execute("INSERT INTO vectors (id, vector) VALUES (?, ?)", (row['reviewId'], vector))

# Commit the changes to the database and close the connection
conn.commit()
conn.close()
