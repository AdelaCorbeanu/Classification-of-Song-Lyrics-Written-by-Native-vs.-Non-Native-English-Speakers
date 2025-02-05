import nltk
import torch
import re
import numpy as np
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm


dataset = load_dataset("AdelaCorbeanu/English-Lyrical-Origins")

nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


def clean_lyrics(lyrics):
    lyrics = re.sub(r'\[.*?\]', '', lyrics)  # remove text inside brackets
    lyrics = lyrics.lower()
    # lyrics = ''.join([char for char in lyrics if char.isalnum() or char.isspace()])  # remove punctuation
    tokens = word_tokenize(lyrics)
    # tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(tokens)


train_lyrics_cleaned = [clean_lyrics(lyric) for lyric in dataset['train']['title']]
test_lyrics_cleaned = [clean_lyrics(lyric) for lyric in dataset['test']['title']]

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(dataset['train']['native'])
test_labels = label_encoder.transform(dataset['test']['native'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=2).to(device)


def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")



class LyricsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenize_function(texts)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]


train_dataset = LyricsDataset(train_lyrics_cleaned, train_labels)
test_dataset = LyricsDataset(test_lyrics_cleaned, test_labels)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch, labels in tqdm(train_loader):
        batch = {key: val.to(device) for key, val in batch.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch, labels in test_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        labels = labels.to(device)
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")