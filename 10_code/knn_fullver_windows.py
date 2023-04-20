import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier

# Load training features and labels
train_features = np.load(r"D:\140Ktrain_images.npy")
train_labels = np.load(r"D:\140Ktrain_labels.npy")

num_samples = train_features.shape[0]
num_features = np.prod(train_features.shape[1:])
train_features_2d = np.reshape(train_features, (num_samples, num_features))

# Load validation features and labels
val_features = np.load(r"D:\140Kval_images.npy")
val_labels = np.load(r"D:\140Kval_labels.npy")

num_samples = val_features.shape[0]
num_features = np.prod(val_features.shape[1:])
val_features_2d = np.reshape(val_features, (num_samples, num_features))

# Load test features and labels
test_features = np.load(r"D:\140Ktest_images.npy")
test_labels = np.load(r"D:\140Ktest_labels.npy")

num_samples = test_features.shape[0]
num_features = np.prod(test_features.shape[1:])
test_features_2d = np.reshape(test_features, (num_samples, num_features))

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

# Fit the classifier to the training data
knn.fit(train_features_2d, train_labels)


# Create a predefined train/test split for RandomSearchCV (to be used later)
X_train_plus_val = np.concatenate((train_features_2d, val_features_2d), axis=0)
y_train_plus_val = np.concatenate((train_labels, val_labels), axis=0)

from sklearn.model_selection import PredefinedSplit

validation_fold = np.concatenate(
    (-1 * np.ones(len(train_labels)), np.zeros(len(val_labels)))
)
train_val_split = PredefinedSplit(validation_fold)

# Run RandomizedSearchCV to find the best hyperparameters
from sklearn.model_selection import RandomizedSearchCV

# define models and parameters
model_2 = KNeighborsClassifier()
n_neighbors = range(1, 5)

param_dist_2 = {"n_neighbors": n_neighbors}

# define random search
n_iter_search_2 = 20
random_search_2 = RandomizedSearchCV(
    model_2,
    param_distributions=param_dist_2,
    n_iter=n_iter_search_2,
    cv=train_val_split,
    random_state=10,
    n_jobs=-1,
)

# from signal import signal, SIGPIPE, SIG_DFL
import time

# signal(SIGPIPE, SIG_DFL)
import os

os.environ["JOBLIB_TEMP_FOLDER"] = r"E:"
start = time.time()
random_search_2.fit(X_train_plus_val, y_train_plus_val)
print(
    "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
    % ((time.time() - start), n_iter_search_2)
)

# save value of best k
best_k = random_search_2.best_params_["n_neighbors"]


# Run KNN with best k with complete training and validation data
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_plus_val, y_train_plus_val, n_jobs=-1)

# Make predictions on the test data
test_predictions = knn.predict(test_features_2d)

# Calculate the accuracy of the predictions
accuracy = np.mean(test_predictions == test_labels)
print("Test Accuracy: ", accuracy)

# ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(test_labels, test_predictions)

roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(
    fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
