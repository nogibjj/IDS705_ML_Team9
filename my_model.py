"""Resnet model for classification of real and fake images."""

import os

import matplotlib.pyplot as plt

import numpy as np

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

from tensorflow.keras.layers import Flatten

from keras.layers.core import Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

real_data_directory = "Data/Real"
fake_data_directory = "Data/Fake"
from PIL import Image

img = Image.open(r"Data\Real\00998.png")

# img.show(), shows the image in viewer

# img.size, returns the size of the image

# img.format, returns the format of the image

# img.mode, returns the mode of the image (RGB)

real_images = []
real_images_labels = []

fake_images = []
fake_images_labels = []

# for i in range(1, len(os.listdir(real_data_directory)) + 1):

#         real_images.append(Image.open((real_data_directory + '/' + str(i) + '.png')))

#         fake_images.append((fake_data_directory + '/' + str(i) + '.png'))

from PIL import Image
import glob

for filename in glob.glob("Data/Real/*.png"):  # assuming png
    im = Image.open(filename)
    real_images.append(im)
    real_images_labels.append(1)


for filename in glob.glob("Data/Fake/*.jpg"):  # assuming jpg
    im = Image.open(filename)
    filename = filename.replace("Fake", "Fakepng")
    im.save((filename[:-3] + "png"))

for filename in glob.glob("Fakepng/*.png"):  # assuming png
    fake_images.append(im)
    fake_images_labels.append(0)

combined_images = real_images + fake_images
combined_labels = real_images_labels + fake_images_labels

# last combine
final_combined_labels = []


fake_counter = 0
for filename in glob.glob("Data/Fakepng/*.png"):  # assuming png
    im = Image.open(filename)
    new_filename = "Data/combined_subset/" + "F" + str(fake_counter) + ".png"
    im.save(new_filename)
    final_combined_labels.append(0)
    fake_counter += 1

real_counter = 0
for filename in glob.glob("Data/Real/*.png"):  # assuming png
    im = Image.open(filename)
    new_filename = "Data/combined_subset/" + "R" + str(real_counter) + ".png"
    im.save(new_filename)
    final_combined_labels.append(1)
    real_counter += 1
    if real_counter == 100:
        break


IMAGE_SIZE = [128, 128]

batch_size = 32

final_combined_labels_array = np.array(final_combined_labels)


# adjustment for tensorflow 2.0

train_ds = tflow.keras.preprocessing.image_dataset_from_directory(
    # "Data/combined_subset",
    "dataforoldtf",
    labels=final_combined_labels,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=batch_size,
)


validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(
    # "Data/combined_subset",
    "dataforoldtf",
    validation_split=0.2,
    subset="validation",
    seed=123,
    labels=final_combined_labels,
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=batch_size,
)
tflow.debugging.set_log_device_placement(True)

print("Num GPUs Available: ", len(tflow.config.list_physical_devices("GPU")))


# Test for gpu code
# Create some tensors
a = tflow.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tflow.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tflow.matmul(a, b)

print(c)


# Visualize six random images from combined list

zipped_images = tuple(zip(combined_images, combined_labels))

import matplotlib.pyplot as plotter_lib

plotter_lib.figure(figsize=(10, 10))

epochs = 10

for images, labels in train_ds.take(1):
    for var in range(6):
        ax = plt.subplot(3, 3, var + 1)

        plotter_lib.imshow(images[var].numpy().astype("uint8"))

        plotter_lib.axis("off")


demo_resnet_model = Sequential()

# testing the cpu time
import time

cpu_time = time.time()
with tflow.device("/CPU:0"):
    # Create some tensors
    a = tflow.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tflow.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tflow.matmul(a, b)
cpu_time = time.time() - cpu_time

print("CPU time: ", cpu_time)

# testing the gpu time

gpu_time = time.time()

with tflow.device("/GPU:0"):
    # Create some tensors

    a = tflow.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    b = tflow.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    c = tflow.matmul(a, b)

gpu_time = time.time() - gpu_time

print("GPU time: ", gpu_time)


pretrained_model_for_demo = tflow.keras.applications.ResNet50(
    include_top=False,
    input_shape=(128, 128, 3),
    pooling="avg",
    classes=5,
    weights="imagenet",
)

for each_layer in pretrained_model_for_demo.layers:
    each_layer.trainable = False

demo_resnet_model.add(pretrained_model_for_demo)


demo_resnet_model.add(Flatten())

demo_resnet_model.add(Dense(512, activation="relu"))

demo_resnet_model.add(Dense(1, activation="sigmoid"))

demo_resnet_model.compile(
    optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"]
)

history = demo_resnet_model.fit(train_ds, validation_data=validation_ds, epochs=epochs)

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

plotter_lib.ylabel("Cross Entropy")

plotter_lib.title("Training and Validation Loss")

plotter_lib.xlabel("epoch")

plotter_lib.show()

# Save the model

demo_resnet_model.save("demo_resnet_model.h5")

# Load the model

demo_resnet_model_saved = tflow.keras.models.load_model("demo_resnet_model.h5")

# Predict on a single image

img = Image.open("Data/combined_subset/F0.png")

img = img.resize((128, 128))

img = np.expand_dims(img, axis=0)

img = np.array(img)

img = img / 255.0

prediction = demo_resnet_model_saved.predict(img)

print(prediction)

# Predict on a batch of images

batch_of_images = []

for filename in glob.glob("Data/combined_subset/*.png"):  # assuming png
    im = Image.open(filename)

    im = im.resize((128, 128))

    im = np.expand_dims(im, axis=0)

    im = np.array(im)

    im = im / 255.0

    batch_of_images.append(im)

batch_of_images = np.array(batch_of_images)

batch_of_images = np.reshape(batch_of_images, (200, 224, 224, 3))

predictions = demo_resnet_model.predict(batch_of_images)

print(predictions)


### From reference

plotter_lib.figure(figsize=(8, 8))

epochs_range = range(epochs)

plotter_lib.plot(epochs_range, history.history["accuracy"], label="Training Accuracy")

plotter_lib.plot(
    epochs_range, history.history["val_accuracy"], label="Validation Accuracy"
)

plotter_lib.axis(ymin=0.4, ymax=1)

plotter_lib.grid()

plotter_lib.title("Model Accuracy")

plotter_lib.ylabel("Accuracy")

plotter_lib.xlabel("Epochs")

plotter_lib.legend(["train", "validation"])

plotter_lib.show()

plotter_lib.savefig("output-plot.png")


# matching dimension images step

import cv2

sample_image = cv2.imread("Data\combined_subset\R64.png")

sample_image_resized = cv2.resize(sample_image, (IMAGE_SIZE))

sample_image = np.expand_dims(sample_image_resized, axis=0)

image_pred = demo_resnet_model.predict(sample_image)

# image_output_class=class_names[np.argmax(image_pred)]

# print("The predicted class is", image_output_class)


# Leveraging prefetching

# AUTOTUNE = tflow.data.experimental.AUTOTUNE
AUTOTUNE = tflow.data.AUTOTUNE


tflow.test.gpu_device_name()

tflow.config.experimental.list_physical_devices("GPU")


# Activating gpu

# gpus = tflow.config.experimental.list_physical_devices("GPU")

# if gpus:

#     try:

#         tflow.config.experimental.set_virtual_device_configuration(

#             gpus[0], [tflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]

#         )

#     except RuntimeError as e:

#         print(e)
