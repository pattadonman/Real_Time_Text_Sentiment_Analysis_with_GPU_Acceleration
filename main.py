
#---------------------------------------------------------------------------------------------------------------------------------
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ตรวจสอบว่า CUDA ใช้ได้หรือไม่
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# แสดงข้อความว่าใช้ GPU หรือ CPU
if torch.cuda.is_available():
    print("Using GPU for computation.")
else:
    print("Using CPU for computation.")

# โหลดโมเดลและ Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# ส่งโมเดลไปยัง GPU หรือ CPU ตามที่มี
model.to(device)

# ตัวอย่างข้อมูล (ข้อความ)
texts = [
    "I love this product!",
    "This is the worst experience I've had.",
    "It's okay, neither good nor bad."
]

# แปลงข้อความให้เป็น token ids
class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return encoding

# สร้าง DataLoader สำหรับข้อมูล
dataset = SentimentDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# การทำนาย
model.eval()
predictions = []

with torch.no_grad():
    for batch in dataloader:
        # ส่งข้อมูลไปยัง GPU หรือ CPU
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        
        # ทำนาย
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)

# สร้างป้ายกำกับสำหรับผลลัพธ์
labels = ['Positive', 'Negative', 'Neutral']

# แสดงผลการทำนาย
for text, prediction in zip(texts, predictions):
    print(f"Text: {text}\nSentiment: {labels[int(prediction)]}\n")

