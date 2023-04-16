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

# Image opening test
# This lets us know that the folders are in optimal condition
# For the first few test runs, we will utilize the One1KsetDraft


# Sanity check for the image opening

try:
    # Using the 10K folder for this case
    Image.open("tenKDataset\Real\Real7998.png")

    print("Image opened successfully")

except:
    print("Image not found")


### Image opening test finished


# Set up the image size, this is the default for ResNet50
IMAGE_SIZE = (256, 256)  # height, width

batch_size = 32  # 32 is default recommendation for vision models


# calculate length  of elements in One1ksetDraft
n_samples = 0
for each_folder in os.listdir("T9-140KRGB"):
    print("Folder: {}".format(each_folder))
    print(
        "Number of images: {}".format(
            len(os.listdir("T9-140KRGB/{}".format(each_folder)))
        )
    )
    n_samples += len(os.listdir("T9-140KRGB/{}".format(each_folder)))


# # the ratio for the splits
# train_ratio = 0.7
# val_ratio = 0.3


# Create the dataset
# adjustment for tensorflow 2.0

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


### Validating Batches ###
# for the train split, this is to validate that
# the batching worked
count = 0
img_gs = []
label_gs = []
for img, label in train_ds.take(-1):
    count += 1
    img_gs.append(img.numpy())
    label_gs.append(label.numpy())
print(f"Number of batches in Train: {count}")


### Validating Batches Complete

### Enhancing reproducibility
train_ds.shuffle(count, reshuffle_each_iteration=False)
del count, img_gs, label_gs
gc.collect()
### Separating the validation set
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


# # for the test split
# val_batches = tflow.data.experimental.cardinality(validation_ds)
# test_ds = validation_ds.take((2 * val_batches) // 3)
# val_ds = validation_ds.skip((2 * val_batches) // 3)

# check for classes in the dataset
# class_names = validation_ds.class_names
v_im = []
v_lab = []
for img, label in validation_ds.take(-1):
    v_im.append(img.numpy())
    v_lab.append(label.numpy())
v_lab_broken = np.array([float(label) for batch in v_lab for label in batch])
print(
    "\n",
    f"For Team 9 : \n\
      Validation Set Labels: \n\
        {np.unique(v_lab_broken, return_counts=True)} \n\
      Class names are: {validation_ds.class_names}",
)
v_lab_string = np.unique(v_lab_broken, return_counts=True)
v_lab_class = validation_ds.class_names
del v_im, v_lab, v_lab_broken
gc.collect()

# check for classes in the test_ds dataset
# class_names = validation_ds.class_names

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

# check for classes in the dataset
t_im = []
t_lab = []
for img, label in test_ds.take(-1):
    t_im.append(img.numpy())
    t_lab.append(label.numpy())
t_lab_broken = np.array([float(label) for batch in t_lab for label in batch])
print(
    "\n",
    f"For Team 9: \n\
      Test Set Labels: \n\
        {np.unique(t_lab_broken, return_counts=True)} \n\
        Class names are: {test_ds.class_names}",
)

test_string = np.unique(t_lab_broken, return_counts=True)
test_classes = test_ds.class_names
del t_im, t_lab, t_lab_broken
gc.collect()


# check for classes in the train_ds dataset
# class_names = validation_ds.class_names
tr_im = []
tr_lab = []
for img, label in train_ds.take(-1):
    tr_im.append(img.numpy())
    tr_lab.append(label.numpy())
tr_lab_broken = np.array([float(label) for batch in tr_lab for label in batch])
print(
    "\n",
    f"For Team 9: \n\
      Train Set Labels: \n\
        {np.unique(tr_lab_broken, return_counts=True)} \n\
        Class names are: {train_ds.class_names}",
)
train_string = np.unique(tr_lab_broken, return_counts=True)
train_classes = train_ds.class_names
del tr_im, tr_lab, tr_lab_broken
gc.collect()

# saving the images and labels for the train, test and validation sets


ims = []
labs = []
probs = []
preds = []

datasets = {"train": train_ds, "test": test_ds, "val": validation_ds}
ds_arrays = {}
for ds in datasets:
    ims = []
    labs = []

    for image, label in datasets[ds]:
        ims.append(image)
        labs.append(label)

    ims = np.concatenate(ims, axis=0)
    labs = np.concatenate(labs, axis=0)
    print(ims.shape)
    print(labs.shape)

    ds_arrays[ds] = {"images": ims, "labels": labs}
del ims, labs
gc.collect()
# save the arrays
for ds in ds_arrays:
    np.save(f"140K{ds}_images.npy", ds_arrays[ds]["images"])
    np.save(f"140K{ds}_labels.npy", ds_arrays[ds]["labels"])

del ds_arrays
gc.collect()

print("Num GPUs Available: ", len(tflow.config.list_physical_devices("GPU")))
print(tflow.config.list_physical_devices("GPU"))
print("Num CPUs Available: ", len(tflow.config.list_physical_devices("CPU")))
print(tflow.config.list_physical_devices("CPU"))


# Visualize six random images from combined list

# import matplotlib.pyplot as plotter_lib

# plotter_lib.figure(figsize=(10, 10))

epochs = 10

# for images, labels in train_ds.take(1):
#     for var in range(6):
#         ax = plt.subplot(3, 3, var + 1)

#         plotter_lib.imshow(images[var].numpy().astype("uint8"))

#         plotter_lib.axis("off")


# Model creation

demo_resnet_model = Sequential()
IMAGE_SIZE_INT = 256
IMAGE_SIZE = (256, 256)  # height, width
pretrained_model_for_demo = tflow.keras.applications.ResNet50(
    include_top=False,
    input_shape=(IMAGE_SIZE_INT, IMAGE_SIZE_INT, 3),
    pooling="avg",
    weights="imagenet",
)

for each_layer in pretrained_model_for_demo.layers:
    each_layer.trainable = False

demo_resnet_model.add(pretrained_model_for_demo)


demo_resnet_model.add(Flatten())

demo_resnet_model.add(Dense(512, activation="relu"))

demo_resnet_model.add(Dense(1, activation="sigmoid"))


demo_resnet_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        "AUC",
        "Precision",
        "Recall",
        "TruePositives",
        "TrueNegatives",
        "FalsePositives",
        "FalseNegatives",
    ],
)

history = demo_resnet_model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    verbose=1,
    shuffle=False,  # For reproducibility
    # callbacks=[tflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
)
model_val_tr_results = history.history

# write the model_val_tr_results to a file
with open("model_val_tr_results.txt", "w") as file:
    file.write(str(model_val_tr_results))


# save the model at this step

final_hp = demo_resnet_model.get_config()

demo_resnet_model.save("demo_resnet_model_20K.h5")

demo_resnet_model.save_weights("demo_resnet_model_weights_20K.h5")


# processing the predictions
imgs_arr_val = []
labels_arr_val = []
pred_arr_val = []
combined_val = []

for img, label in val_ds.take(-1):
    imgs_arr_val.append(img.numpy())
    labels_arr_val.append(label.numpy())
    pred_val = demo_resnet_model.predict(img)
    pred_arr_val.append(pred_val)
    combined_val.append((img.numpy(), label.numpy(), pred_val))

val_label_final = np.array([])
val_pred_final = np.array([])
val_counter = 0

for im_val, label_val, pred_val in combined_val:

    val_label_final = np.append(val_label_final, label_val)
    val_pred_final = np.append(val_pred_final, pred_val)

    # Sanity checking the misclassifications
    misclassifieds = label_val - np.round((pred_val))
    val_counter += np.sum((abs(misclassifieds)))

print(f"Number of misclassified images in the validation set: {val_counter}")


y = np.concatenate([y for x, y in val_ds], axis=0)
x = np.concatenate([x for x, y in val_ds], axis=0)
pred = demo_resnet_model.predict(x)

# imagenes =np.array([])
# labels = np.array([])
# predictions_probas = np.array([])
# predictions = np.array([])
ims = []
labs = []
probs = []
preds = []

ls = []
n = 3000
while n != 0:
    ims = []
    labs = []
    probs = []
    preds = []
    for image, label in val_ds:
        # print(image.shape, label.shape)
        # imagenes = np.concatenate((imagenes, image), axis=0)
        # labels = np.concatenate((labels, label), axis=0)
        # predictions_probas = np.concatenate((predictions_probas, demo_resnet_model.predict(image)), axis=0)
        # predictions = np.concatenate((predictions, np.round(demo_resnet_model.predict(image))), axis=0)
        ims.append(image)
        labs.append(label)
        probs.append(demo_resnet_model.predict(image))
        preds.append(np.round(demo_resnet_model.predict(image)))

    custom_dict = {"ims": ims, "labs": labs, "probs": probs, "preds": preds}
    new_dict = {}
    for key in custom_dict:

        new_dict[key] = np.concatenate(custom_dict[key], axis=0)

    my_val = np.sum(abs((new_dict["labs"] - np.round(new_dict["preds"]))))
    ls.append(my_val)
    n -= 1

ls.index(min(ls))


from sklearn.metrics import roc_auc_score, roc_curve, auc

tflow.data.datasets.from_tensor_slices(val_ds)
fpr, tpr, thresholds = roc_curve(val_label_final, val_pred_final)
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
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
# save the ROC curve
plt.show()
plt.savefig("ROC_curve.png")


# Visualize the training and validation accuracy and loss

plotter_lib.figure(figsize=(10, 10))

plotter_lib.subplot(2, 1, 1)

plotter_lib.plot(history.history["accuracy"], label="Training Accuracy")

plotter_lib.plot(history.history["val_accuracy"], label="Validation Accuracy")

plotter_lib.legend(loc="lower right")

plotter_lib.ylabel("Accuracy")

plotter_lib.title("Training and Validation Accuracy")

plotter_lib.subplot(2, 1, 2)

plotter_lib.plot(history.history["loss"], label="Training Loss")

plotter_lib.plot(history.history["val_loss"], label="Validation Loss")

plotter_lib.legend(loc="upper right")

plotter_lib.ylabel("Binary Cross Entropy")

plotter_lib.title("Training and Validation Loss")

plotter_lib.xlabel("epoch")

plotter_lib.show()


from sklearn.metrics import classification_report

cs_report_val = classification_report(val_label_final, val_pred_final.round())


# Training on the entire dataset


demo_resnet_model_fin = Sequential()
IMAGE_SIZE_INT = 256
IMAGE_SIZE = (256, 256)  # height, width
pretrained_model_for_demo_fin = tflow.keras.applications.ResNet50(
    include_top=False,
    input_shape=(IMAGE_SIZE_INT, IMAGE_SIZE_INT, 3),
    pooling="avg",
    weights="imagenet",
)

for each_layer in pretrained_model_for_demo_fin.layers:
    each_layer.trainable = False

demo_resnet_model_fin.add(pretrained_model_for_demo_fin)

demo_resnet_model_fin.add(Flatten())

demo_resnet_model_fin.add(Dense(512, activation="relu"))

demo_resnet_model_fin.add(Dense(1, activation="sigmoid"))

demo_resnet_model_fin.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        "AUC",
        "Precision",
        "Recall",
        "TruePositives",
        "TrueNegatives",
        "FalsePositives",
        "FalseNegatives",
    ],
)

train_ds_and_val_ds = train_ds.concatenate(val_ds)
history2 = demo_resnet_model_fin.fit(
    train_ds_and_val_ds,
    epochs=epochs,
    verbose=1,
    validation_data=None,
    shuffle=False,  # For reproducibility
    # callbacks=[tflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
)

# Save the model
final_hp = demo_resnet_model_fin.get_config()

demo_resnet_model.save("demo_resnet_model_20K.h5")

demo_resnet_model.save_weights("demo_resnet_model_weights_20K.h5")


# zip the saved models
import zipfile

with zipfile.ZipFile("demo_resnet_model_20K.zip", "w") as zip:
    zip.write(
        "demo_resnet_model_20K.h5", compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
    )
    zip.write(
        "demo_resnet_model_weights_20K.h5",
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )


# Visualize the training and validation accuracy and loss

plotter_lib.figure(figsize=(10, 10))

plotter_lib.subplot(2, 1, 1)

plotter_lib.plot(history.history["accuracy"], label="Training Accuracy")

plotter_lib.plot(history.history["val_accuracy"], label="Validation Accuracy")

plotter_lib.legend(loc="lower right")

plotter_lib.ylabel("Accuracy")

plotter_lib.title("Training and Validation Accuracy")

plotter_lib.subplot(2, 1, 2)

plotter_lib.plot(history.history["loss"], label="Training Loss")

plotter_lib.plot(history.history["val_loss"], label="Validation Loss")

plotter_lib.legend(loc="upper right")

plotter_lib.ylabel("Binary Cross Entropy")

plotter_lib.title("Training and Validation Loss")

plotter_lib.xlabel("epoch")

plotter_lib.show()

# save the plot
plotter_lib.savefig("Atraining_and_validation_accuracy_and_loss.png")
plotter_lib.savefig(
    "Btraining_and_validation_accuracy_and_loss.png", dpi=300, bbox_inches="tight"
)
