"""Resnet model for classification of real and fake images."""

import os

import matplotlib.pyplot as plt

import numpy as np

import glob

import PIL as image_lib

import tensorflow as tflow

from tensorflow.keras.layers import Flatten

from keras.layers.core import Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from PIL import Image

import matplotlib.pyplot as plotter_lib

import numpy as np

import PIL as image_lib

import tensorflow as tflow

from tensorflow.keras.metrics import Accuracy

from tensorflow.keras.layers import Flatten


from keras.layers.core import Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

import time

import gc

from PIL import Image

# Timing the script
script_time = time.time()


# Apparently for monitoring, it only shows in terminals
tflow.debugging.set_log_device_placement(True)


# Set up Prefetching
AUTOTUNE = tflow.data.AUTOTUNE


# Set up the image size, this is the default for ResNet50
IMAGE_SIZE = (256, 256)  # height, width

batch_size = 128  # 32 is default recommendation for vision models


train_ds = tflow.keras.preprocessing.image_dataset_from_directory(
    "T9-Train",
    label_mode="binary",
    # validation_split=val_ratio,
    shuffle=True,
    # subset="training",
    seed=417,
    image_size=IMAGE_SIZE,
    batch_size=batch_size,  # Changed from batch_size 32 to none
    # color_mode="grayscale",
)

validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(
    "T9-Val",
    # validation_split=val_ratio,
    shuffle=True,
    # subset="validation",
    seed=417,
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=batch_size,  # Changed from batch_size 32 to none
    # color_mode="grayscale",
)

train_ds_and_val_ds = train_ds.concatenate(validation_ds)


### Validating Batches ###
# for the train split, this is to validate that
# the batching worked
count = 0
img_gs = []
label_gs = []
for img, label in train_ds_and_val_ds.take(-1):
    count += 1
    img_gs.append(img.numpy())
    label_gs.append(label.numpy())
print(f"Number of batches in TrainplusVal: {count}")


train_ds_and_val_ds.shuffle(count, reshuffle_each_iteration=False)
del count, img_gs, label_gs
gc.collect()

# test dataset
test_ds = tflow.keras.preprocessing.image_dataset_from_directory(
    "T9-Test",
    label_mode="binary",
    shuffle=True,
    seed=417,
    image_size=IMAGE_SIZE,
    batch_size=batch_size,  # Changed from batch_size 32 to none
    # color_mode="grayscale",
)

def class_dist(ds, name):
    """Returns a dictionary of class distribution."""
    t_im = []
    t_lab = []
    for img, label in ds.take(-1):
        t_im.append(img.numpy())
        t_lab.append(label.numpy())
    t_lab_broken = np.array([float(label) for batch in t_lab for label in batch])
    print(
        "\n",
        f"For Team 9: \n\
        {name} Set Labels: \n\
            {np.unique(t_lab_broken, return_counts=True)} \n\
            Class names are: {ds.class_names}")

    return ds.class_names

epochs = 10


# Model creation


# Load the model

demo_resnet_model = tflow.keras.models.load_model("demo_resnet_model_140K.h5")
demo_resnet_model.summary()
demo_resnet_model.get_config()
demo_resnet_model.get_weights()
demo_resnet_model.optimizer
demo_resnet_model.loss
demo_resnet_model.metrics
# Our model has been validated

history = demo_resnet_model.fit(
    train_ds_and_val_ds,
    # validation_data=validation_ds,
    epochs=epochs,
    verbose=1,
    shuffle=False,  # For reproducibility
    # callbacks=[tflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
)
# Agreed, GPU was the only option for this model
# This is 30 minutes on a 10 epoch run
# Approximately 28K images in Test set

# Saving the model fitted on the train and validation set

demo_resnet_model.save("demo_resnet_model_140K_tr_val_rgb.h5")
demo_resnet_model.save_weights("demo_resnet_model_140K_tr_val_rgb_weights.h5")

demo_resnet_model = tflow.keras.models.load_model("demo_resnet_model_140K_tr_val_rgb.h5")
# checkpoint
# demo_resnet_model = tflow.keras.models.load_model("demo_resnet_model_140K_tr_val_rgb.h5")
# demo_resnet_model = tflow.keras.models.load_weights("demo_resnet_model_140K_tr_val_rgb_weights.h5")

# below code works for the dataset
# because it is neither concatenated nor type changed
ims = []
labs = []
probs = []
preds = []
for image, label in test_ds:
    ims.append(image)
    labs.append(label)
    probs.append(demo_resnet_model.predict(image))
    preds.append(np.round(demo_resnet_model.predict(image)))

ims = np.concatenate(ims, axis=0)
labs = np.concatenate(labs, axis=0)
probs = np.concatenate(probs, axis=0)
preds = np.concatenate(preds, axis=0)

misclassifieds = np.where(labs != preds)[0]

misc_ims = ims[misclassifieds]
misc_labs = labs[misclassifieds]
im_labels = zip(misc_ims, misc_labs)
iter_for_name = 0
for im, lab in im_labels:
    if lab == 0:
        flag = "Fake"
    else:
        flag = "Real"

    reconstructed_image = Image.fromarray((im * 1).astype(np.uint8)).convert("RGB")

    # use old ind for naming
    reconstructed_image.save(f"T9-Misc140KRGB\{flag}\{flag}{iter_for_name}.png")
    iter_for_name += 1


print(f"Number of misclassified images in the test set: {len(misc_labs)}")


### Add normal Evaluation code here ###

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(labs, preds))
print(confusion_matrix(labs, preds))
# write classification report to file
with open("test_classification_report.txt", "w") as f:
    f.write(str(classification_report(labs, preds)))
    f.close()

# write confusion matrix to file
with open("test_confusion_matrix.txt", "w") as f:
    f.write(str(confusion_matrix(labs, preds)))
    f.close()


evaluation = demo_resnet_model.evaluate(test_ds, verbose=1, return_dict=True)
print(evaluation)
# write evaluation to file
with open("test_evaluation.txt", "w") as f:
    f.write(str(evaluation))
    f.write(str(test_ds.class_names))
    f.close()

### Add PR and ROC code here ###

import matplotlib.pyplot as plotter_lib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import numpy as np


def roc_curve_T9(labels_ls, test_ds_pred):
    fpr, tpr, _ = roc_curve(labels_ls, test_ds_pred)
    auc_score = auc(fpr, tpr)
    # plot roc curve
    plotter_lib.plot(
        fpr,
        tpr,
        color="orange",
        label=f"ResNet50 (AUC = {auc_score:.4f})",
    )
    # axis labels
    # add random guessing line
    plotter_lib.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plotter_lib.xlabel("False Positive Rate")
    plotter_lib.ylabel("True Positive rate")
    plotter_lib.legend(loc="best")
    plotter_lib.title("ROC Curve on Test Set")
    plotter_lib.show()


roc_curve_T9(labs, probs)


def pr_curve_T9(labels_ls, test_ds_pred):
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(labels_ls[labels_ls == 1]) / len(labels_ls)
    # calculate model precision-recall curve
    precision, recall, _ = precision_recall_curve(labels_ls, test_ds_pred)
    # average precision score
    ap = average_precision_score(labels_ls, test_ds_pred)
    # plot the precision-recall curves
    plotter_lib.plot(
        [0, 1], [no_skill, no_skill], linestyle="--", label="Random Chance"
    )
    plotter_lib.plot(recall, precision, label="ResNet50 (AP = %.4f)" % ap)
    # axis labels
    plotter_lib.xlabel("Recall")
    plotter_lib.ylabel("Precision")
    plotter_lib.title("Precision-Recall Curve on Test Set")
    plotter_lib.legend()
    plotter_lib.show()


pr_curve_T9(labs, probs)

demo_resnet_model.save("demo_resnet_model_140K_tr_val_rgb.h5")
demo_resnet_model.save_weights("demo_resnet_model_140K_tr_val_rgb_weights.h5")

f1_score(labs, preds)

# Final Step : Train on all data and save model

all_ds = train_ds_and_val_ds.concatenate(test_ds)
# validating the number of images in the dataset
count = 0
for im, label in all_ds.take(-1):
    count += im.shape[0]

print("The images in the dataset with all the data are: ", count)

# Train on all data

final_history = demo_resnet_model.fit(
    all_ds,
    # validation_data=validation_ds,
    epochs=epochs,
    verbose=1,
    shuffle=False,  # For reproducibility
    callbacks=[tflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
)

# save the final model because Team 9 is the best!!!!

demo_resnet_model.save("deploy_resnet_model_140K_alldata_rgb.h5")
demo_resnet_model.save_weights("deploy_resnet_model_140K_alldata_rgb_weights.h5")


####################
# Proxy work is left to another script
####################


# Proxy : AI generated images


demo_resnet_model_proxy = tflow.keras.models.load_model("deploy_resnet_model_140K_alldata_rgb.h5")



for image in os.listdir(r"Data\Application\very_real_ai-faces\very_real_ai-faces"):
    img = tflow.keras.preprocessing.image.load_img(
        r"Data\Application\very_real_ai-faces\very_real_ai-faces\{}".format(
            image
        ),
        target_size=(256, 256),
    )
    img_array = tflow.keras.preprocessing.image.img_to_array(img)
    prediction_proxy = demo_resnet_model.predict(img_array)
    label = np.round(prediction_proxy)
    if label == 0:
        flag = "Correct"
    else:
        flag = "Incorrect"
    reconstructed_image = Image.fromarray((img_array * 1).astype(np.uint8)).convert(
        "RGB"
    )

    reconstructed_image.save(f"AIProxy\{flag}\{image}.png")

    # img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # predictions = demo_resnet_model.predict(img_array)
    # score = tf.nn.softmax(predictions[0])

    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
    #         class_names[np.argmax(score)], 100 * np.max(score)
    #     )
    # )

# Proxy MIDS people

for image in os.listdir(
    r"C:\Users\Eric\Downloads\MLprojectamd\MIDS24-20230415T234818Z-001\MIDS24"
):
    img = tflow.keras.preprocessing.image.load_img(
        r"C:\Users\Eric\Downloads\MLprojectamd\MIDS24-20230415T234818Z-001\MIDS24\{}".format(
            image
        ),
        target_size=(256, 256),
    )
    img_array = tflow.keras.preprocessing.image.img_to_array(img)

    prediction_proxy = demo_resnet_model.predict(img_array)
    label = np.round(prediction_proxy)
    if label == 0:
        flag = "SUCKERS"
    else:
        flag = "BORING"
    reconstructed_image = Image.fromarray((img_array * 1).astype(np.uint8)).convert(
        "RGB"
    )

    reconstructed_image.save(f"MIDSFUN\{flag}\{image}.png")


# Proxy : Dating App Part 1
# This needs to be edited further, it's a different proxy
for image in os.listdir(r" "):
    img = tflow.keras.preprocessing.image.load_img(
        r" ".format(image),
        target_size=(256, 256),
    )
    img_array = tflow.keras.preprocessing.image.img_to_array(img)
    prediction_proxy = demo_resnet_model.predict(img_array)
    label = np.round(prediction_proxy)
    if label == 0:
        flag = "Fake"
    else:
        flag = "Real"
    reconstructed_image = Image.fromarray((img_array * 1).astype(np.uint8)).convert(
        "RGB"
    )

    reconstructed_image.save(f"DatingApp\{flag}\{image}.png")


script_time = time.time() - script_time
print("Total GPU script time: {} seconds".format(script_time))
print("We did it, everybody. See you next time!")
print("The full train, val and test run has completed")
print("Test Metrics for the ResNet50 model")
print(f"Test metrics : {evaluation}")
print("Kashaf, Bella, Eric, and mean old Tensorflow : GPU Version")
# write the results to a file
with open("output.txt", "w") as f:
    f.write("Total GPU script time: {} seconds".format(script_time))
    f.write("The full train, val and test run has completed")
    f.write("Test Metrics for the ResNet50 model")
    f.write(f"Test metrics : {evaluation}")
