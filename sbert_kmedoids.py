import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
import re
import numpy as np
import string
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from itertools import product
from sklearn.model_selection import train_test_split

dataset = load_dataset("AdelaCorbeanu/English-Lyrical-Origins")

nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


def clean_lyrics(lyrics):
    # remove text inside brackets
    lyrics = re.sub(r'\[.*?\]', '', lyrics)

    lyrics = lyrics.lower()  

    # remove punctuation (characters that are neither space, nor alpha-numeric)
    lyrics = ''.join([char for char in lyrics if char.isalnum() or char.isspace()])

    # remove stopwords
    tokens = word_tokenize(lyrics)
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)


train_lyrics_cleaned = [clean_lyrics(lyric) for lyric in dataset['train']['lyrics']]
test_lyrics_cleaned = [clean_lyrics(lyric) for lyric in dataset['test']['lyrics']]

train_labels = dataset['train']['native']
test_labels = dataset['test']['native']

# convert string labels to numbers
label_encoder = LabelEncoder()
train_true_labels = label_encoder.fit_transform(train_labels)
test_true_labels = label_encoder.transform(test_labels)

# extract numerical features using Sentence-BERT
sbert_model = SentenceTransformer('princeton-nlp/unsup-simcse-bert-base-uncased')
X_train = sbert_model.encode(train_lyrics_cleaned, convert_to_numpy=True)
X_test = sbert_model.encode(test_lyrics_cleaned, convert_to_numpy=True)

# extract a small validation subset from the training data
X_train, X_val, y_train, y_val = train_test_split(X_train, train_true_labels, test_size=0.1)

# hyperparameters values to try
pca_components = [5, 10, 20]
n_clusters_list = [2, 3, 4]
metrics_list = ['euclidean', 'manhattan', 'cosine', 'chebyshev', 'correlation']
best_model = None
best_ari = -1

results = []

for n_components, n_clusters, metric in product(pca_components, n_clusters_list, metrics_list):
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train)
    X_val_reduced = pca.transform(X_val)

    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric)
    train_clusters = kmedoids.fit_predict(X_train_reduced)
    val_clusters = kmedoids.predict(X_val_reduced)

    # evaluate performance on validation data
    train_ari_score = adjusted_rand_score(y_train, train_clusters)
    val_ari_score = adjusted_rand_score(y_val, val_clusters)
    train_silhouette = silhouette_score(X_train_reduced, train_clusters)
    val_silhouette = silhouette_score(X_val_reduced, val_clusters)

    results.append({
        'PCA Components': n_components,
        'Clusters': n_clusters,
	'Metric': metric,
        'Train ARI': train_ari_score,
        'Val ARI': val_ari_score,
        'Train Silhouette': train_silhouette,
        'Val Silhouette': val_silhouette
    })

    if val_ari_score > best_ari:
        best_ari = val_ari_score
        best_model = (n_components, n_clusters, kmedoids, pca, metric)


for result in results:
    print(f"PCA Components: {result['PCA Components']}, Clusters: {result['Clusters']}, Metric: {result['Metric']}")
    print(f"Train ARI: {result['Train ARI']:.2f}, Val ARI: {result['Val ARI']:.2f}")
    print(f"Train Silhouette: {result['Train Silhouette']:.2f}, Val Silhouette: {result['Val Silhouette']:.2f}\n")


# evaluate the best model on the test data
best_pca, best_kmedoids = best_model[3], best_model[2]

X_train_reduced = best_pca.transform(X_train)
X_test_reduced = best_pca.transform(X_test)

train_clusters = best_kmedoids.fit_predict(X_train_reduced)
test_clusters = best_kmedoids.predict(X_test_reduced)

test_ari_score = adjusted_rand_score(test_true_labels, test_clusters)
test_silhouette = silhouette_score(X_test_reduced, test_clusters)

print(f"Best Model - PCA Components: {best_model[0]}, Clusters: {best_model[1]}, Metric: {best_model[4]}")
print(f"Test ARI: {test_ari_score:.2f}, Test Silhouette: {test_silhouette:.2f}")

