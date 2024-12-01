import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset


data = pd.read_csv('Preprocessing/movie1000.csv')
data=data.head(1000)
relevantData = data[['reviewText','scoreSentiment']]
relevantData = relevantData.dropna(subset=['reviewText', 'scoreSentiment'])
print(data.isnull().sum())

# Encode sentiment labels
sentiment_mapping = {'POSITIVE': 0, 'NEGATIVE': 1, 'NEUTRAL': 2}
relevantData['scoreSentiment'] = relevantData['scoreSentiment'].map(sentiment_mapping)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Compute token lengths for each review
relevantData['token_lengths'] = relevantData['reviewText'].apply(lambda x: len(tokenizer.tokenize(x)))

#summary statistics
print(relevantData['token_lengths'].describe())

# Percentiles (90th, 95th)
print(relevantData['token_lengths'].quantile([0.9, 0.95]))


plt.hist(relevantData['token_lengths'], bins=30, alpha=0.7, color='blue')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.title('Token Length Distribution')
plt.show()

def tokenize_function(text, tokenizer, max_length=52):
    tokens = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',  # Pad to max_length
        return_tensors="pt"    # Return as PyTorch tensors
    )
    return tokens['input_ids'], tokens['attention_mask']

max_length = 52  # Adjust based on dataset
relevantData['input_ids'], relevantData['attention_mask'] = zip(*relevantData['reviewText'].apply(
    lambda x: tokenize_function(x, tokenizer, max_length)
))

# First, split into training + test and validation sets (80% train + 20% validation/test)
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    relevantData[['input_ids', 'attention_mask']].values,
    relevantData['scoreSentiment'].values,
    test_size=0.2,  # 80% for train + validation, 20% for test
    stratify=relevantData['scoreSentiment'],
    random_state=42
)

# Then, split the training+validation set into separate training and validation sets (80% train, 20% validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts,
    train_val_labels,
    test_size=0.2,  # 80% for train, 20% for validation
    stratify=train_val_labels,
    random_state=42
)

# Ensure that train_texts, val_texts, and test_texts are tensors or NumPy arrays
train_inputs = torch.stack([item[0] for item in train_texts], dim=0).long()  # stack tensors to create a single tensor
train_masks = torch.stack([item[1] for item in train_texts], dim=0).long()   # stack attention_masks
train_labels = torch.tensor(train_labels, dtype=torch.long)

val_inputs = torch.stack([item[0] for item in val_texts], dim=0).long()
val_masks = torch.stack([item[1] for item in val_texts], dim=0).long()
val_labels = torch.tensor(val_labels, dtype=torch.long)

test_inputs = torch.stack([item[0] for item in test_texts], dim=0).long()
test_masks = torch.stack([item[1] for item in test_texts], dim=0).long()
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Create DataLoader objects
batch_size = 16
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Save the datasets (train, validation, and test)
torch.save(train_data, 'train_data.pt')
torch.save(val_data, 'val_data.pt')
torch.save(test_data, 'test_data.pt')