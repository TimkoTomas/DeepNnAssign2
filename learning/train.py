import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

# Load the CSV file
df = pd.read_csv('movie1000.csv')

# Preprocess the 'reviewText' column (e.g., remove NaNs, lowercasing, etc.)
df['reviewText'] = df['reviewText'].fillna('').astype(str).str.lower()

# Encode sentiment labels (assuming 'POSITIVE' -> 1 and 'NEGATIVE' -> 0)
df['label'] = df['scoreSentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

# Split the dataset into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['reviewText', 'label']])
eval_dataset = Dataset.from_pandas(eval_df[['reviewText', 'label']])

# Tokenize the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['reviewText'], truncation=True, padding=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fine-tuned-bert')
tokenizer.save_pretrained('./fine-tuned-bert')