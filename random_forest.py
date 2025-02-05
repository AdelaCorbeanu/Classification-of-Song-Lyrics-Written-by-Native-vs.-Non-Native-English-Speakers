import nltk
import re
import numpy as np
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

dataset = load_dataset("AdelaCorbeanu/English-Lyrical-Origins")

nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

def clean_lyrics(lyrics):
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
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

rf_model = RandomForestClassifier(
    n_estimators=200,         
    max_depth=20,             
    min_samples_split=10,     
    min_samples_leaf=2,       
    max_features='sqrt',      
    random_state=42           
)

rf_model.fit(X_train, train_labels)

predictions = rf_model.predict(X_test)
accuracy = accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")



feature_importances = rf_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()

top_n = 20
top_indices = np.argsort(feature_importances)[::-1][:top_n]
top_ngrams = [(feature_names[i], feature_importances[i]) for i in top_indices]

print("\nTop 20 Most Relevant Character n-grams:")
for ngram, importance in top_ngrams:
    print(f"{ngram}: {importance:.6f}")
