import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Đọc dữ liệu từ file CSV
file_path = '/home/huyhoa/PycharmProjects/nhom_12/data/fake_and_real_news_dataset.csv'  # Thay bằng đường dẫn tới file của bạn
df = pd.read_csv(file_path)

# 1. Trực quan hóa sự phân bố của các label (REAL hoặc những label khác)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('label_distribution.png')
plt.close()

# 2. Vẽ biểu đồ độ dài văn bản
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(df['text_length'], kde=True)
plt.title('Distribution of Text Length')
plt.xlabel('Text Length (characters)')
plt.ylabel('Frequency')
plt.savefig('text_length_distribution.png')
plt.close()

# 3. Tạo word cloud từ cột 'text'
text_combined = " ".join(df['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Text Data')
plt.savefig('wordcloud.png')
plt.close()

print("All visualizations have been saved as images.")
