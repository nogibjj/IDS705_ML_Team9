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


# the ratio for the splits
train_ratio = 0.7
val_ratio = 0.3


# Create the dataset
# adjustment for tensorflow 2.0

train_ds = tflow.keras.preprocessing.image_dataset_from_directory(
    "tenKDataset",
    label_mode="binary",
    validation_split=val_ratio,
    shuffle=True,
    subset="training",
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

### Separating the validation set
validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(
    "tenKDataset",
    validation_split=val_ratio,
    shuffle=True,
    subset="validation",
    seed=417,
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=batch_size,  # Changed from batch_size 32 to none
    # color_mode="grayscale",
)


# for the test split
val_batches = tflow.data.experimental.cardinality(validation_ds)
test_ds = validation_ds.take((2 * val_batches) // 3)
val_ds = validation_ds.skip((2 * val_batches) // 3)

# check for classes in the dataset
# class_names = validation_ds.class_names
v_im = []
v_lab = []
for img, label in val_ds.take(-1):
    v_im.append(img.numpy())
    v_lab.append(label.numpy())
v_lab_broken = np.array([float(label) for batch in v_lab for label in batch])
print(
    "\n",
    f"For Team 9 : \n\
      Validation Set Labels: \n\
        {np.unique(v_lab_broken, return_counts=True)} \n\
      Class names are: {train_ds.class_names}",
)

np.unique(v_lab_broken, return_counts=True)

# check for classes in the test_ds dataset
# class_names = validation_ds.class_names
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
        Class names are: {train_ds.class_names}",
)

np.unique(t_lab_broken, return_counts=True)

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
np.unique(tr_lab_broken, return_counts=True)

# import python garbage collector
import gc

del v_im, v_lab, t_im, t_lab, tr_im, tr_lab

gc.collect()

# saving the images and labels for the train, test and validation sets


ims = []
labs = []
probs = []
preds = []

datasets = {"train": train_ds, "test": test_ds, "val": val_ds}
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

# save the arrays
for ds in ds_arrays:
    np.save(f"140K{ds}_images.npy", ds_arrays[ds]["images"])
    np.save(f"140K{ds}_labels.npy", ds_arrays[ds]["labels"])


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
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
    shuffle=False,  # For reproducibility
    # callbacks=[tflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
)
model_val_tr_results = history.history

# write the model_val_tr_results to a file
with open("model_val_tr_results.txt", "w") as file:
    file.write(str(model_val_tr_results))

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

    custom_dict = {'ims': ims, 'labs': labs, 'probs': probs, 'preds': preds}
    new_dict = {}
    for key in custom_dict:

        new_dict[key] = np.concatenate(custom_dict[key], axis=0)

    my_val = np.sum(abs((new_dict['labs'] - np.round(new_dict['preds']))))
    ls.append(my_val)
    n -= 1

ls.index(min(ls))


from sklearn.metrics import roc_auc_score, roc_curve, auc
tflow.data.datasets.from_tensor_slices(val_ds)
fpr, tpr, thresholds = roc_curve(val_label_final, val_pred_final)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
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

demo_resnet_model_fin.save("demo_resnet_model_20K.h5")

demo_resnet_model_fin.save_weights("demo_resnet_model_weights_20K.h5")


# zip the saved models
import zipfile

with zipfile.ZipFile("demo_resnet_model_20K.zip", "w") as zip:
    zip.write("demo_resnet_model_20K.h5", compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    zip.write("demo_resnet_model_weights_20K.h5")




###### End Script Here - Use main.py to run the script ######
############################################################################################################


# Load the model

# demo_resnet_model_fin = tflow.keras.models.load_model("demo_resnet_model.h5")

# predicting on the training set
prediction_time = time.time()

# train_predictions = demo_resnet_model.predict(train_ds)

test_ds_pred = demo_resnet_model_fin.predict(test_ds)

prediction_time = time.time() - prediction_time

print("Prediction time for test set: {} seconds".format(prediction_time))
test_evaluation = demo_resnet_model.evaluate(test_ds, verbose=1, return_dict=True)

# write test out_evaluation to file
with open("test_evaluation.txt", "w") as f:

    for key, value in test_evaluation.items():

        f.write("%s:%s

# To do list:
    # full script
    # partition to test script
    # aucs
    # basic hp tuning
    # 140K run
    # model saving
    # experiment wiht reloading a deactivated shuffle


test_ds.labels


# processing the predictions
imgs_arr = []
labels_arr = []
# labels_arr = np.array([])
combined = []

for img, label in test_ds.take(-1):
    # print(img.dtype)
    imgs_arr.append(img.numpy())
    labels_arr.append(label.numpy())
    # pred = np.round(demo_resnet_model.predict(img))
    pred = demo_resnet_model.predict(img)
    combined.append((img.numpy(), label.numpy(), pred))
    # labels_arr = np.append(labels_arr, label.numpy())

imgs_ls = [img for batch in imgs_arr for img in batch]
labels_ls = np.array([label for batch in labels_arr for label in batch])

counter_other = 0

bad_classifications = {}
batch_num = 0

# MAJOR PROGRESS FOUND HERE
#### BAD IMAGE ISOLATION ######
for imagenes, etiquetas, predicciones in combined:
    # counter_other += np.sum((etiquetas != np.round((predicciones))))

    misclassifieds = etiquetas - np.round((predicciones))
    counter_other += np.sum((abs(misclassifieds)))

    # bad_classifications.append(
    #     np.where((misclassifieds > 0), misclassifieds)
    # )
    # misclassifieds
    target = []
    j = 0
    for i in misclassifieds:
        print(i)
        if i == 1:
            target.append(j)

        elif i == -1:
            target.append(-j)

        j += 1
    bad_classifications[batch_num] = target
    batch_num += 1

    # print("ok")
# I had issues importing my own functions,
# so I just copied the code here
import glob


def erase_images(output_dir, format="jpg"):
    """Erase images in output directory"""
    for image in glob.glob(os.path.join(output_dir, ("*." + format))):
        os.remove(image)


erase_images("Misclassifications\Real", format="png")
erase_images("Misclassifications\Fake", format="png")
# Don't trust the numbers of the pictures, they
# are true to the batch only
# Isolations of bad images
for key, value in bad_classifications.items():
    misclassifieds = bad_classifications[
        key
    ]  # tuple of lists of indices of misclassified images

    for ind in misclassifieds:
        print(ind)
        if ind < 0:
            flag = "Fake"

            new_ind = abs(ind)

        else:
            flag = "Real"
            new_ind = ind

        image_array = combined[key][0][new_ind]

        reconstructed_image = Image.fromarray(
            (image_array * 1).astype(np.uint8)
        ).convert("RGB")

        # use old ind for naming
        reconstructed_image.save(f"Misclassifications\{flag}\{flag}{ind+1}.png")


###### BAD IMAGE ISOLATION ######


# labels_ls shares the same dimensions as test_ds_pred
# now we round
# test_ds_pred_rounded = np.round(test_ds_pred)
test_ds_pred_other_class = 1 - test_ds_pred
test_ds_pred_rounded = np.round(test_ds_pred_other_class)
np.sum(test_ds_pred_rounded == labels_ls) / len(labels_ls)

# why does this gave me such a low accuracy?
# accuracy = np.sum(test_ds_pred_rounded == labels_ls) / len(labels_ls)

# Example of image reconstruction
# Image.fromarray((imgs_ls[0] * 255).astype(np.uint8)).convert("RGB")
# solution
# Image.fromarray((imgs_ls[0] * 1).astype(np.uint8)).convert("RGB")

# roc curve for models
from sklearn.metrics import roc_curve

fpr, tpr, thresh = roc_curve(labels_ls, test_ds_pred)
# plot roc curve
plotter_lib.plot(fpr, tpr, linestyle="--", color="orange", label="ResNet50")
# axis labels
plotter_lib.xlabel("False Positive Rate")
plotter_lib.ylabel("True Positive rate")
plotter_lib.legend(loc="best")
plotter_lib.title("ROC curve")
plotter_lib.show()


from sklearn.metrics import classification_report
classification_report(labels_ls, test_ds_pred.round())


# Predict on a single image

# single_image_prediction_time = time.time()

# img = Image.open("Data/combined_subset/F0.png")

# img = img.resize((128, 128))

# img = np.expand_dims(img, axis=0)

# img = np.array(img)

# img = img / 255.0

# prediction = demo_resnet_model.predict(img)

# single_image_prediction_time = time.time() - single_image_prediction_time

# print(prediction)

# print(
#     "Prediction time for single image: {} seconds".format(single_image_prediction_time)
# )


# Predict on a batch of images

# batch_of_images = []
# batch_prediction_time = time.time()
# for filename in glob.glob("Data/combined_subset/*.png"):  # assuming png
#     im = Image.open(filename)

#     im = im.resize((128, 128))

#     im = np.expand_dims(im, axis=0)

#     im = np.array(im)

#     im = im / 255.0

#     batch_of_images.append(im)

# batch_of_images = np.array(batch_of_images)

# batch_of_images = np.reshape(batch_of_images, (200, 128, 128, 3))

# predictions = demo_resnet_model.predict(batch_of_images)
# batch_prediction_time = time.time() - batch_prediction_time
# # print(predictions)
# print("Prediction time for batch of images: {} seconds".format(batch_prediction_time))

### From reference

# plotter_lib.figure(figsize=(8, 8))

# epochs_range = range(epochs)

# plotter_lib.plot(epochs_range, history.history["accuracy"], label="Training Accuracy")

# plotter_lib.plot(
#     epochs_range, history.history["val_accuracy"], label="Validation Accuracy"
# )

# plotter_lib.axis(ymin=0.4, ymax=1)

# plotter_lib.grid()

# plotter_lib.title("Model Accuracy")

# plotter_lib.ylabel("Accuracy")

# plotter_lib.xlabel("Epochs")

# plotter_lib.legend(["train", "validation"])

# plotter_lib.show()

# plotter_lib.savefig("output-plot.png")

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
