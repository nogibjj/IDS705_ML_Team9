import matplotlib.pyplot as plotter_lib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import numpy as np

def roc_curve(labels_ls, test_ds_pred):
    test_ds_pred_other_class = 1 - test_ds_pred
    test_ds_pred_rounded = np.round(test_ds_pred_other_class)
    np.sum(test_ds_pred_rounded == labels_ls) / len(labels_ls)
    fpr, tpr, thresh = roc_curve(labels_ls, test_ds_pred)
    # plot roc curve
    plotter_lib.plot(fpr, tpr, linestyle="--", color="orange", label="ResNet50")
    # axis labels
    plotter_lib.xlabel("False Positive Rate")
    plotter_lib.ylabel("True Positive rate")
    plotter_lib.legend(loc="best")
    plotter_lib.title("ROC curve")
    plotter_lib.show()

def pr_curve(labels_ls, test_ds_pred):
    # calculate the no skill line as the proportion of the positive class
    fake = len(labels_ls[labels_ls==1]) / len(labels_ls)
    # calculate model precision-recall curve
    precision, recall, _ = precision_recall_curve(labels_ls, test_ds_pred)
    # plot the precision-recall curves
    plotter_lib.plot([0, 1], [fake, fake], linestyle='--', label='No Skill')
    plotter_lib.plot(recall, precision, marker='.', label='ResNet50')
    # axis labels
    plotter_lib.xlabel('Recall')
    plotter_lib.ylabel('Precision')
    plotter_lib.legend()
    plotter_lib.show()

# generate fake data
labels_ls = np.random.randint(0, 2, 100)
test_ds_pred = np.random.rand(100)

# plot roc curve
roc_curve(labels_ls, test_ds_pred)

# plot pr curve
pr_curve(labels_ls, test_ds_pred)