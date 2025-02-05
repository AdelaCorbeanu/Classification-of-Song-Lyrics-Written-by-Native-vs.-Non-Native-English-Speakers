import nltk
import re
import numpy as np
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


dataset = load_dataset("AdelaCorbeanu/English-Lyrical-Origins")

nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


def clean_lyrics(lyrics):
    lyrics = re.sub(r'\[.*?\]', '', lyrics)  # remove text inside brackets
    tokens = word_tokenize(lyrics)
    return ' '.join(tokens)


train_lyrics_cleaned = [clean_lyrics(lyric) for lyric in dataset['train']['lyrics']]
test_lyrics_cleaned = [clean_lyrics(lyric) for lyric in dataset['test']['lyrics']]

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(dataset['train']['native'])
test_labels = label_encoder.transform(dataset['test']['native'])

vectorizer = TfidfVectorizer(ngram_range=(1,7), analyzer='char')  
X_train = vectorizer.fit_transform(train_lyrics_cleaned)
X_test = vectorizer.transform(test_lyrics_cleaned)

svm_model = SVC()
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
}

grid_search = GridSearchCV(svm_model, param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, train_labels)

best_svm_model = grid_search.best_estimator_
predictions = best_svm_model.predict(X_test)

accuracy = accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")
