
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

# Đảm bảo bạn sử dụng đúng đường dẫn tới mô hình đã lưu
model_path = 'model/fine_tune_bert'  # Thay đổi nếu cần

# Tải lại mô hình đã được huấn luyện
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Đảm bảo test_dataset đã được định dạng đúng
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1, pin_memory=True)

# Đánh giá trên tập kiểm tra
y_pred = []
model.eval()  # Chuyển sang chế độ đánh giá

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        y_pred.extend(predictions.cpu().numpy())

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print('Final Test Set Accuracy:', accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
# Lưu kết quả vào tệp 'evaluate_bert.txt'
with open("evaluate_bert.txt", "w") as file:
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Classification Report:\n{classification_report(y_test, y_pred)}")


# Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=.5, square=True, cmap='Blues_r', fmt='d')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - Final Model', size=15)
plt.savefig("confusion_matrix_saved_model.png")
plt.show()
