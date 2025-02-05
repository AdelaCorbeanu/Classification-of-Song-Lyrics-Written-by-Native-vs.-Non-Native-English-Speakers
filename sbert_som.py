import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
import re
import numpy as np
import string
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from itertools import product
from minisom import MiniSom

dataset = load_dataset("AdelaCorbeanu/English-Lyrical-Origins")

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


def clean_lyrics(lyrics):
    # remove text inside brackets
    lyrics = re.sub(r'\[.*?\]', '', lyrics)

    lyrics = lyrics.lower()  

    # remove punctuation (characters that are neither space, nor alpha-numeric)
    lyrics = [''.join([char for char in text if char.isalnum() or char.isspace()]) for text in lyrics]  

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

np.save("X_train_embeddings.npy", X_train)
np.save("X_test_embeddings.npy", X_test)




import numpy as np
import joblib
from itertools import product
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from minisom import MiniSom
from datasets import load_dataset

dataset = load_dataset("AdelaCorbeanu/English-Lyrical-Origins")

# load previously saved embeddings
X_train = np.load("X_train_embeddings.npy")
X_test = np.load("X_test_embeddings.npy")

label_encoder = LabelEncoder()
train_labels = dataset["train"]["native"]
test_labels = dataset["test"]["native"]
train_true_labels = label_encoder.fit_transform(train_labels)
test_true_labels = label_encoder.transform(test_labels)

# extract a small validation subset from the training data
X_train, X_val, y_train, y_val = train_test_split(X_train, train_true_labels, test_size=0.1, random_state=42)

# hyperparameter values to try
pca_components = [5, 6, 7, 8, 15, 9, 10, 15, 20, 50, 100, 500]
som_sizes = [(1, 2), (2, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
learning_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
sigmas = [0.1, 0.5, 1.0, 2.0, 2.5, 3.0]
best_model = None
best_ari = -1
num_iter = 1

results = []

for n_components, (som_x, som_y), learning_rate, sigma in product(pca_components, som_sizes, learning_rates, sigmas):
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train)
    X_val_reduced = pca.transform(X_val)

    for num_iter in [100, 500, 1000, 5000, 10000]:
      som = MiniSom(som_x, som_y, input_len=n_components, learning_rate=learning_rate, sigma=sigma)
      som.train_random(X_train_reduced, num_iter)

      train_clusters = np.array([som.winner(x) for x in X_train_reduced])
      val_clusters = np.array([som.winner(x) for x in X_val_reduced])

      # convert 2D coordinates to 1D
      train_clusters = np.ravel_multi_index(np.array(train_clusters).T, (som_x, som_y))
      val_clusters = np.ravel_multi_index(np.array(val_clusters).T, (som_x, som_y))

      train_ari_score = adjusted_rand_score(y_train, train_clusters)
      val_ari_score = adjusted_rand_score(y_val, val_clusters)
      train_silhouette = silhouette_score(X_train_reduced, train_clusters)
      val_silhouette = silhouette_score(X_val_reduced, val_clusters)

      results.append({
          "PCA Components": n_components,
          "SOM Size": (som_x, som_y),
          "Learning Rate": learning_rate,
          "Sigma": sigma,
          "Num Iterations": num_iter,
          "Train ARI": train_ari_score,
          "Val ARI": val_ari_score,
          "Train Silhouette": train_silhouette,
          "Val Silhouette": val_silhouette,
      })

      # track best model
      if val_ari_score > best_ari:
          best_ari = val_ari_score
          best_model = (n_components, (som_x, som_y), som, pca, learning_rate, sigma, num_iter)

for result in results:
    print(f"PCA Components: {result['PCA Components']}, SOM Size: {result['SOM Size']}, Learning Rate: {result['Learning Rate']}, Sigma: {result['Sigma']}, Iterations: {result['Num Iterations']}")
    print(f"Train ARI: {result['Train ARI']:.2f}, Val ARI: {result['Val ARI']:.2f}")
    print(f"Train Silhouette: {result['Train Silhouette']:.2f}, Val Silhouette: {result['Val Silhouette']:.2f}\n")

best_pca = best_model[3]
best_som = best_model[2]
num_iter = best_model[6]

X_train_reduced = best_pca.transform(X_train)
X_test_reduced = best_pca.transform(X_test)

best_som.train_random(X_train_reduced, num_iter)


def get_cluster_distribution(clusters, labels):
    cluster_distribution = {}
    for cluster in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        cluster_labels = labels[cluster_indices]  # Actual labels (0 = non-native, 1 = native)

        native_count = np.sum(cluster_labels == 1)
        non_native_count = np.sum(cluster_labels == 0)

        cluster_distribution[cluster] = {'native': native_count, 'non_native': non_native_count}

    return cluster_distribution


train_clusters = np.array([best_som.winner(x) for x in X_train_reduced])
train_clusters = np.ravel_multi_index(np.array(train_clusters).T, best_model[1])

train_cluster_distribution = get_cluster_distribution(train_clusters, y_train)

print("Train Cluster Distributions:")
for cluster, counts in train_cluster_distribution.items():
    print(f"Cluster {cluster}: Native = {counts['native']}, Non-Native = {counts['non_native']}")

test_clusters = np.array([best_som.winner(x) for x in X_test_reduced])
test_clusters = np.ravel_multi_index(np.array(test_clusters).T, best_model[1])

test_cluster_distribution = get_cluster_distribution(test_clusters, test_true_labels)

print("Test Cluster Distributions:")
for cluster, counts in test_cluster_distribution.items():
    print(f"Cluster {cluster}: Native = {counts['native']}, Non-Native = {counts['non_native']}")