import matplotlib.pyplot as plotter_lib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np


def roc_curve_T9(labels_ls, test_ds_pred):
    # test_ds_pred_other_class = 1 - test_ds_pred
    # test_ds_pred_rounded = np.round(test_ds_pred_other_class)
    # np.sum(test_ds_pred_rounded == labels_ls) / len(labels_ls)
    fpr, tpr, thresh = roc_curve(labels_ls, test_ds_pred)
    auc_score = auc(fpr, tpr)
    # plot roc curve
    plotter_lib.plot(
        fpr,
        tpr,
        linestyle="--",
        color="orange",
        label=f"ResNet50 (AUC = {auc_score:.3f})",
    )
    # axis labels
    plotter_lib.xlabel("False Positive Rate")
    plotter_lib.ylabel("True Positive rate")
    plotter_lib.legend(loc="best")
    plotter_lib.title("ROC Curve on Test Set")
    plotter_lib.show()


def pr_curve_T9(labels_ls, test_ds_pred):
    # calculate the no skill line as the proportion of the positive class
    fake = len(labels_ls[labels_ls == 1]) / len(labels_ls)
    # calculate model precision-recall curve
    precision, recall, _ = precision_recall_curve(labels_ls, test_ds_pred)
    # plot the precision-recall curves
    plotter_lib.plot([0, 1], [fake, fake], linestyle="--", label="Random Chance")
    plotter_lib.plot(recall, precision, marker=".", label="ResNet50")
    # axis labels
    plotter_lib.xlabel("Recall")
    plotter_lib.ylabel("Precision")
    plotter_lib.legend()
    plotter_lib.show()


def auc(labels_ls, test_ds_pred):
    # calculate the no skill line as the proportion of the positive class
    fake = len(labels_ls[labels_ls == 1]) / len(labels_ls)
    # calculate model precision-recall curve
    precision, recall, _ = precision_recall_curve(labels_ls, test_ds_pred)
    # add random chance line
    plotter_lib.plot([0, 1], [fake, fake], linestyle="--", label="Random Chance")
    # calculate AUC
    auc = auc(recall, precision)
    print("AUC: %.3f" % auc)


if __name__ == "__main__":
    # generate fake data
    labels_ls = np.random.randint(0, 2, 100)
    test_ds_pred = np.random.rand(100)

    # plot roc curve
    roc_curve_T9(labels_ls, test_ds_pred)

    # plot pr curve
    pr_curve_T9(labels_ls, test_ds_pred)
