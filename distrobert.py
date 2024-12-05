import pandas as pd
import src.config
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dữ liệu
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['title'] = df['title'].fillna('')  # Thay thế NaN bằng chuỗi rỗng
    df['text'] = df['text'].fillna('')  # Thay thế NaN bằng chuỗi rỗng
    df['combined_text'] = df['title'] + " " + df['text']  # Kết hợp 'title' và 'text'

    df['label'] = df['label'].apply(lambda x: 0 if x == 'FAKE' else 1)  # Đổi nhãn thành 0 và 1
    df = df[['combined_text', 'label']]
    return df


data = load_data(src.config.DATA_PATH)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['combined_text'], data['label'], test_size=0.2, random_state=42
)

# Tạo tập Dataset cho HuggingFace
train_dataset = Dataset.from_dict({'combined_text': train_texts, 'label': train_labels})
test_dataset = Dataset.from_dict({'combined_text': test_texts, 'label': test_labels})


from transformers import AutoTokenizer

# Tải tokenizer của DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Token hóa dữ liệu
def tokenize_function(batch):
    return tokenizer(batch['combined_text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Loại bỏ cột 'text', chỉ giữ lại input_ids, attention_mask và label
train_dataset = train_dataset.remove_columns(["combined_text"])
test_dataset = test_dataset.remove_columns(["combined_text"])

# Đổi tên nhãn để phù hợp với PyTorch format
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

# Chuyển đổi sang định dạng PyTorch
train_dataset.set_format("torch")
test_dataset.set_format("torch")
print(len(train_dataset))
print(len(test_dataset))
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Tải mô hình DistilBERT
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)
# Thiết lập tham số huấn luyện
training_args = TrainingArguments(
    output_dir=src.config.OUTPUT_DIR,          # Thư mục lưu kết quả
    evaluation_strategy="epoch",    # Đánh giá sau mỗi epoch
    save_strategy="epoch",
    learning_rate=src.config.LEARNING_RATE,
    per_device_train_batch_size=16, # Batch size
    per_device_eval_batch_size=16,
    num_train_epochs=3,             # Số epoch
    weight_decay=0.01,              # L2 regularization
    logging_dir=src.config.LOGGING_DIR,           # Thư mục lưu log
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Hàm compute_metrics tính toán độ chính xác
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=1)  # Lấy chỉ số lớp có xác suất cao nhất
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

# Tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Huấn luyện
trainer.train()


def save_model(model, tokenizer, output_dir, model_name="distrobert-pretrain"):
    """
    Lưu mô hình và tokenizer đã được huấn luyện với tên tùy chỉnh.

    Args:
        model (BertForSequenceClassification): Mô hình đã được huấn luyện.
        tokenizer (BertTokenizer): Tokenizer đã được sử dụng.
        output_dir (str): Thư mục lưu trữ.
        model_name (str): Tên của mô hình được lưu.
    """
    # Thư mục con để lưu mô hình với tên tùy chỉnh
    save_path = f"{output_dir}/{model_name}"

    # Lưu mô hình và tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Mô hình và tokenizer đã được lưu tại {save_path}")

save_model(model, tokenizer, src.config.OUTPUT_DIR)

from sklearn.metrics import accuracy_score, classification_report

# Dự đoán trên tập test
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)
y_true = test_dataset['labels']

# Đánh giá hiệu năng
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_true, y_pred))
# Lưu kết quả vào tệp 'evaluate_distrobert.txt'
with open("evaluate_distrobert.txt", "w") as file:
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Classification Report:\n{classification_report(y_true, y_pred)}")


def create_dir(dir_path):
    """
    Tạo thư mục nếu chưa tồn tại.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# Tạo confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
eval_dir=src.config.OUTPUT_DIR

# Lưu kết quả dưới dạng CSV
create_dir(eval_dir)
class_report_df = pd.DataFrame(classification_report).transpose()

# Thêm accuracy vào DataFrame
class_report_df.loc['accuracy'] = ['-', '-', accuracy, class_report_df['support'].sum()]

# Lưu kết quả dưới dạng CSV
csv_path = os.path.join(eval_dir, 'evaluation_results_distro_bert.csv')
class_report_df.to_csv(csv_path)
print(f"Lưu kết quả dưới dạng CSV tại {csv_path}")

# Vẽ và lưu confusion matrix
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(eval_dir, 'confusion_matrix_distro_bert.png'))
plt.close()
print(f"Lưu confusion matrix tại {eval_dir}/confusion_matrix.png")
