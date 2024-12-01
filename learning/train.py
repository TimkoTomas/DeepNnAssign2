import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import os

# Load pre-saved datasets
train_data = torch.load('train_data.pt')
val_data = torch.load('val_data.pt')
test_data = torch.load('test_data.pt')

# Extract inputs, attention masks, and labels from the TensorDataset
train_inputs, train_masks, train_labels = train_data.tensors
val_inputs, val_masks, val_labels = val_data.tensors
test_inputs, test_masks, test_labels = test_data.tensors

# Remove the extra dimension (squeeze the tensors)
train_inputs = train_inputs.squeeze(1)  # Shape: [582, 52]
train_masks = train_masks.squeeze(1)    # Shape: [582, 52]
val_inputs = val_inputs.squeeze(1)      # Shape: [X, 52]
val_masks = val_masks.squeeze(1)        # Shape: [X, 52]
test_inputs = test_inputs.squeeze(1)    # Shape: [X, 52]
test_masks = test_masks.squeeze(1)      # Shape: [X, 52]

# Print input shapes for debugging
print("Train inputs shape:", train_inputs.shape)
print("Train masks shape:", train_masks.shape)
print("Train labels shape:", train_labels.shape)

# Custom Dataset for Hugging Face Trainer
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.input_ids)

# Create datasets
train_dataset = SentimentDataset(train_inputs, train_masks, train_labels)
val_dataset = SentimentDataset(val_inputs, val_masks, val_labels)
test_dataset = SentimentDataset(test_inputs, test_masks, test_labels)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # num_labels=3 for POSITIVE, NEGATIVE, NEUTRAL

# Function to compute metrics: precision, recall, f1, and accuracy
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)  # Convert logits to predicted labels
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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

# Define Trainer with the compute_metrics function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # Add the compute_metrics function here
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fine-tuned-bert')

# Collect metrics during training (after training is completed)
metrics = trainer.state.log_history

# Filter out the metrics for each epoch
training_metrics = []
for entry in metrics:
    if 'eval_accuracy' in entry:  # These are evaluation metrics (you can adjust to include other metrics)
        training_metrics.append({
            'epoch': entry['epoch'],
            'eval_loss': entry.get('eval_loss', None),
            'eval_accuracy': entry['eval_accuracy'],
            'eval_precision': entry.get('eval_precision', None),
            'eval_recall': entry.get('eval_recall', None),
            'eval_f1': entry.get('eval_f1', None)
        })

# Save metrics to a JSON file in the 'results' folder
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

metrics_file = os.path.join(results_dir, 'training_metrics.json')
with open(metrics_file, 'w') as f:
    json.dump(training_metrics, f, indent=4)

print(f"Training metrics saved to {metrics_file}")
