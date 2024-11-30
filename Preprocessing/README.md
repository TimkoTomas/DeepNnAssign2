### Documentation for Sentiment Analysis Data Preprocessing Code

#### Overview
This script preprocesses a movie review dataset to prepare it for training a sentiment analysis model using BERT. The process includes data loading, cleaning, sentiment label encoding, text tokenization, and splitting the dataset into training, validation, and test sets. It also includes generating statistical analysis on token lengths and visualizing the distribution of token lengths. Finally, the processed data is saved as PyTorch tensors for easy access during model training.

---

### Steps Involved

#### 1. **Data Loading and Cleaning**
- The dataset is loaded from a CSV file (`movie.csv`) using `pandas`. The first 1000 rows are selected for processing to limit the data size for this example.
- The relevant columns (`reviewText` and `scoreSentiment`) are extracted and any rows with missing values in these columns are dropped.
- The presence of missing values is confirmed by printing out the count of null values in each column using `data.isnull().sum()`.

```python
data = pd.read_csv('./movie.csv')
data = data.head(1000)
relevantData = data[['reviewText', 'scoreSentiment']]
relevantData = relevantData.dropna(subset=['reviewText', 'scoreSentiment'])
print(data.isnull().sum())
```

#### 2. **Sentiment Label Encoding**
- The `scoreSentiment` column contains categorical labels (`POSITIVE`, `NEGATIVE`, `NEUTRAL`). These are mapped to numerical values:
  - `POSITIVE` → 0
  - `NEGATIVE` → 1
  - `NEUTRAL` → 2
- This is achieved by using `pandas`' `map()` function.

```python
sentiment_mapping = {'POSITIVE': 0, 'NEGATIVE': 1, 'NEUTRAL': 2}
relevantData['scoreSentiment'] = relevantData['scoreSentiment'].map(sentiment_mapping)
```

#### 3. **Tokenization and Token Length Calculation**
- The `BertTokenizer` from the Hugging Face `transformers` library is used to tokenize the `reviewText` column. Tokenization splits each review text into tokens that can be fed into the BERT model.
- The token lengths (number of tokens per review) are computed using `tokenizer.tokenize(x)` and stored in the new column `token_lengths`.
- Basic statistical analysis on token lengths (e.g., mean, standard deviation) and percentiles (90th, 95th) is generated. A histogram is plotted to visualize the distribution of token lengths.

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
relevantData['token_lengths'] = relevantData['reviewText'].apply(lambda x: len(tokenizer.tokenize(x)))
print(relevantData['token_lengths'].describe())
print(relevantData['token_lengths'].quantile([0.9, 0.95]))
plt.hist(relevantData['token_lengths'], bins=30, alpha=0.7, color='blue')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.title('Token Length Distribution')
plt.show()
```

#### 4. **Text Tokenization with BERT**
- The `tokenize_function` is defined to tokenize the text and pad/truncate each review to a fixed length (`max_length=52` tokens). This ensures that all reviews have the same number of tokens, which is required for BERT.
- The function uses the `BertTokenizer` to convert the text into token IDs (`input_ids`) and the attention mask (`attention_mask`), which tells the model which tokens are padding and which are actual data.

```python
def tokenize_function(text, tokenizer, max_length=52):
    tokens = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',  # Pad to max_length
        return_tensors="pt"    # Return as PyTorch tensors
    )
    return tokens['input_ids'], tokens['attention_mask']
```

#### 5. **Splitting the Dataset**
- The dataset is split into three subsets: training, validation, and test sets.
  - The initial split is 80% for training and validation, and 20% for the test set.
  - The training set is further split into training (80%) and validation (20%) sets.
- This is done using `train_test_split` from `sklearn`, with stratification to maintain the distribution of sentiment labels across all splits.

```python
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    relevantData[['input_ids', 'attention_mask']].values,
    relevantData['scoreSentiment'].values,
    test_size=0.2,
    stratify=relevantData['scoreSentiment'],
    random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts,
    train_val_labels,
    test_size=0.2,
    stratify=train_val_labels,
    random_state=42
)
```

#### 6. **Converting Data to Tensors**
- The `input_ids`, `attention_mask`, and `scoreSentiment` labels are converted into PyTorch tensors, which are the required format for feeding into the BERT model.
- The tokenized `input_ids` and `attention_mask` are stacked into tensors, and the labels are converted to long integer type tensors.

```python
train_inputs = torch.stack([item[0] for item in train_texts], dim=0).long()
train_masks = torch.stack([item[1] for item in train_texts], dim=0).long()
train_labels = torch.tensor(train_labels, dtype=torch.long)

val_inputs = torch.stack([item[0] for item in val_texts], dim=0).long()
val_masks = torch.stack([item[1] for item in val_texts], dim=0).long()
val_labels = torch.tensor(val_labels, dtype=torch.long)

test_inputs = torch.stack([item[0] for item in test_texts], dim=0).long()
test_masks = torch.stack([item[1] for item in test_texts], dim=0).long()
test_labels = torch.tensor(test_labels, dtype=torch.long)
```

#### 7. **Creating DataLoader Objects**
- `DataLoader` objects are created for each dataset (train, validation, and test) using `TensorDataset` to combine the input tensors and labels. These `DataLoader` objects are used to efficiently batch the data during model training.
- The training data is shuffled, while the validation and test data are not.

```python
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=16)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=16)
```

#### 8. **Saving the Data**
- The processed training, validation, and test datasets are saved to disk as `.pt` files using `torch.save()`. These files contain the tokenized inputs, attention masks, and labels, making it easy to load them for future model training.

```python
torch.save(train_data, 'train_data.pt')
torch.save(val_data, 'val_data.pt')
torch.save(test_data, 'test_data.pt')
```

---

### Summary
This script performs the following tasks:
1. Loads and cleans a dataset of movie reviews.
2. Encodes sentiment labels (`POSITIVE`, `NEGATIVE`, `NEUTRAL`) into numerical values.
3. Tokenizes the review text using the BERT tokenizer and computes token lengths.
4. Splits the data into training, validation, and test sets.
5. Converts tokenized data into PyTorch tensors and creates `DataLoader` objects.
6. Saves the processed data into `.pt` files for later use in model training.

