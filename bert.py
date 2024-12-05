import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

# Fine-tuning BERT for Fake News Detection

# Load data
df = pd.read_csv('data/fake_and_real_news_dataset.csv')  # Adjust the path accordingly
df = df.dropna(subset=['text', 'label'])  # Drop rows where 'text' or 'label' are NaN

# Shuffle data
df = shuffle(df).reset_index(drop=True)


# Map labels to integers (0 = FAKE, 1 = REAL)
label_mapping = {'FAKE': 0, 'REAL': 1}
df['label'] = df['label'].map(label_mapping)

# Split data into 64% for training, 16% for validation, 20% for testing
train_set, temp_test_set = train_test_split(df, test_size=0.2, random_state=42)  # 80% train + 20% temp test
val_set, test_set = train_test_split(temp_test_set, test_size=0.5, random_state=42)  # 50% validation + 50% test (of the 20%)

# Split into features (X) and labels (y)
X_train, y_train = train_set['text'], train_set['label']
X_val, y_val = val_set['text'], val_set['label']
X_test, y_test = test_set['text'], test_set['label']

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
val_dataset = Dataset.from_dict({'text': X_val, 'label': y_val})
test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

# Tokenize using BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns for PyTorch training
train_dataset = train_dataset.remove_columns(['text'])
val_dataset = val_dataset.remove_columns(['text'])
test_dataset = test_dataset.remove_columns(['text'])
train_dataset = train_dataset.rename_column('label', 'labels')
val_dataset = val_dataset.rename_column('label', 'labels')
test_dataset = test_dataset.rename_column('label', 'labels')


train_dataset.set_format('torch')
val_dataset.set_format('torch')
test_dataset.set_format('torch')

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Ensure model and datasets are moved to the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training arguments for Hugging Face Trainer
training_args = TrainingArguments(
    output_dir='./model',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,  # Mixed Precision Training
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print('Evaluation Results:', eval_results)

# Final evaluation on test set
y_pred = []
model.eval()
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1, pin_memory=True)

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        y_pred.extend(predictions.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Final Test Set Accuracy:', accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=.5, square=True, cmap='Blues_r', fmt='d')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - Final Model', size=15)
plt.show()

# Classification report
report = classification_report(y_test, y_pred)
print(report)

# Save evaluation model to file
with open('python_bert_848.txt', 'w') as f:
    f.write(f"Evaluation Results:\n{eval_results}\n")
    f.write(f"Final Test Set Accuracy: {accuracy}\n")
    f.write(f"Classification Report:\n{report}\n")

# Optionally, save confusion matrix as an image file
confusion_matrix_path = './confusion_matrix_final_model 848.png'
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=.5, square=True, cmap='Blues_r', fmt='d')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - Final Model', size=15)
plt.savefig(confusion_matrix_path)
plt.close()

# Print confirmation of saved files
print(f"Evaluation model saved to 'python_bert.txt'")
print(f"Confusion matrix saved to '{confusion_matrix_path}'")
