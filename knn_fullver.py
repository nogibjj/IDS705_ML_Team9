import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

# Load training features and labels
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')


# Train KNN model
k = 5  # number of neighbors to consider
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(train_features, train_labels)