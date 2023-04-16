import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier

# Load training features and labels
train_features = np.load(
    "/Users/kashafali/Documents/Duke/Spring23/ML/Project/20karrrays/20Ktrain_images.npy"
)
train_labels = np.load(
    "/Users/kashafali/Documents/Duke/Spring23/ML/Project/20karrrays/20Ktrain_labels.npy"
)

num_samples = train_features.shape[0]
num_features = np.prod(train_features.shape[1:])
train_features_2d = np.reshape(train_features, (num_samples, num_features))

# Load validation features and labels
val_features = np.load(
    "/Users/kashafali/Documents/Duke/Spring23/ML/Project/20karrrays/20Kval_images.npy"
)
val_labels = np.load(
    "/Users/kashafali/Documents/Duke/Spring23/ML/Project/20karrrays/20Kval_labels.npy"
)

num_samples = val_features.shape[0]
num_features = np.prod(val_features.shape[1:])
val_features_2d = np.reshape(val_features, (num_samples, num_features))

# Load test features and labels
test_features = np.load(
    "/Users/kashafali/Documents/Duke/Spring23/ML/Project/20karrrays/20Ktest_images.npy"
)
test_labels = np.load(
    "/Users/kashafali/Documents/Duke/Spring23/ML/Project/20karrrays/20Ktest_labels.npy"
)

num_samples = test_features.shape[0]
num_features = np.prod(test_features.shape[1:])
test_features_2d = np.reshape(test_features, (num_samples, num_features))

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(train_features_2d, train_labels)

# Make predictions on the validation data
val_predictions = knn.predict(val_features_2d)

# Calculate the accuracy of the predictions
accuracy = np.mean(val_predictions == val_labels)
print("Validation Accuracy: ", accuracy)

# Create ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(val_labels, val_predictions)
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
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()
