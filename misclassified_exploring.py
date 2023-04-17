import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tarfile
from mpl_toolkits.mplot3d import Axes3D
import glob
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# Load the image data
real_face_path = '/workspaces/IDS705_ML_Team9/main_model_test_misclassified/Real/*.png'

# read in the first 100 images
real_img_lst = []

for filename in glob.glob(real_face_path):
    with Image.open(filename) as img:
        # convert to jpg
        img = img.convert('RGB')
        # resize the image
        # img = img.resize((128, 128))
        # resize to 256 x 256
        # img = img.resize((256, 256))
        real_img_lst.append(img)

# check the length of the list
len(real_img_lst)


# Load the image data
fake_face_path = '/workspaces/IDS705_ML_Team9/main_model_test_misclassified/Fake/*.png'

# read in the first 100 images
fake_img_lst = []

for filename in glob.glob(fake_face_path):
    with Image.open(filename) as img:
        # convert to jpg
        img = img.convert('RGB')
        # resize the image
        # img = img.resize((128, 128))
        # resize to 256 x 256
        # img = img.resize((256, 256))
        fake_img_lst.append(img)

# check the length of the list
len(fake_img_lst)


# check the resolution of the images
for img in real_img_lst:
    print(f"Image resolution: {img.size}")


# check the resolution of the images
for img in fake_img_lst:
    print(f"Image resolution: {img.size}")


import random

#Real images - Visualise random 30 images
selected_files_true = random.sample(real_img_lst, 30)
# selected_files_true
# visualize the randomly selected images:30 for true images
fig, ax = plt.subplots(5, 6, figsize=(20, 20))

for i in range(5):
    for j in range(6):
        ax[i, j].imshow(selected_files_true[i*6 + j])
        ax[i, j].axis('off')

plt.show()

# save the figure
fig.savefig('real_images.png', dpi=300, bbox_inches='tight')

#Fale images - Visualise random 30 images
selected_files_true = random.sample(fake_img_lst, 30)
# selected_files_true
# visualize the randomly selected images:30 for true images
fig, ax = plt.subplots(5, 6, figsize=(20, 20))

for i in range(5):
    for j in range(6):
        ax[i, j].imshow(selected_files_true[i*6 + j])
        ax[i, j].axis('off')

plt.show()

# save the figure
fig.savefig('fake_images.png', dpi=300, bbox_inches='tight')

# get mean pixel values for real images in a 3 channel format in a 3D plot
real_mean_pixel_values = []

for img in real_img_lst:
    real_mean_pixel_values.append(np.mean(img, axis=(0, 1)))

real_mean_pixel_values = np.array(real_mean_pixel_values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(real_mean_pixel_values[:, 0],
           real_mean_pixel_values[:, 1], real_mean_pixel_values[:, 2])
plt.title('Mean Pixel Values for Misclassified Real Images')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
fig.set_size_inches(10, 10)
plt.show()

# save the figure
fig.savefig('real_misc_mean_pixel_values.png', dpi=300, bbox_inches='tight')

# get mean pixel values for fake images in a 3 channel format in a 3D plot
fake_mean_pixel_values = []

for img in fake_img_lst:
    fake_mean_pixel_values.append(np.mean(img, axis=(0, 1)))

fake_mean_pixel_values = np.array(fake_mean_pixel_values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(fake_mean_pixel_values[:, 0],
           fake_mean_pixel_values[:, 1], fake_mean_pixel_values[:, 2])
plt.title('Mean Pixel Values for Misclassified Fake Images')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
fig.set_size_inches(10, 10)
plt.show()

# save the figure
fig.savefig('fake_misc_mean_pixel_values.png', dpi=300, bbox_inches='tight')


# Initialize empty arrays to hold pixel values for each channel
r_arr_real = np.array([])
g_arr_real = np.array([])
b_arr_real = np.array([])

# Loop over each image in the list of fake image paths
for img_path_r in real_img_lst:
    # Open the image using the PIL Image class
    img_r = img_path_r
    # Split the image into its red, green, and blue channels
    r_r, g_r, b_r = img_r.split()
    # Convert each channel to a NumPy array
    r_arr_img_r = np.array(r_r)
    g_arr_img_r = np.array(g_r)
    b_arr_img_r = np.array(b_r)
    # Append the arrays for each channel to the corresponding global array
    r_arr_real = np.concatenate((r_arr_real, r_arr_img_r.flatten()))
    g_arr_real = np.concatenate((g_arr_real, g_arr_img_r.flatten()))
    b_arr_real = np.concatenate((b_arr_real, b_arr_img_r.flatten()))

# Calculate the mean pixel values for each channel
r_mean_real = np.mean(r_arr_real)
g_mean_real = np.mean(g_arr_real)
b_mean_real = np.mean(b_arr_real)

# get a list of median values for each channel for the real images
r_med_real = np.median(r_arr_real)
g_med_real = np.median(g_arr_real)
b_med_real = np.median(b_arr_real)


# Initialize empty arrays to hold pixel values for each channel
r_arr_fake = np.array([])
g_arr_fake = np.array([])
b_arr_fake = np.array([])

# Loop over each image in the list of fake image paths
for img_path_f in fake_img_lst:
    # Open the image using the PIL Image class
    img_f = img_path_f
    # Split the image into its red, green, and blue channels
    r_f, g_f, b_f = img_f.split()
    # Convert each channel to a NumPy array
    r_arr_img_f = np.array(r_f)
    g_arr_img_f = np.array(g_f)
    b_arr_img_f = np.array(b_f)
    # Append the arrays for each channel to the corresponding global array
    r_arr_fake = np.concatenate((r_arr_fake, r_arr_img_f.flatten()))
    g_arr_fake = np.concatenate((g_arr_fake, g_arr_img_f.flatten()))
    b_arr_fake = np.concatenate((b_arr_fake, b_arr_img_f.flatten()))

# Calculate the mean pixel values for each channel
r_mean_fake = np.mean(r_arr_fake)
g_mean_fake = np.mean(g_arr_fake)
b_mean_fake = np.mean(b_arr_fake)

# get a list of median values for each channel for the fake images
r_med_fake = np.median(r_arr_fake)
g_med_fake = np.median(g_arr_fake)
b_med_fake = np.median(b_arr_fake)




# plot the median values for each channel for the fake images and real images in a 1 by 3 subplot
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax[0].hist(r_arr_fake, bins=50, alpha=0.5, label='Fake')
ax[0].axvline(r_med_fake, color='red', label='Median')

ax[0].hist(r_arr_real, bins=50, alpha=0.5, label='Real')
ax[0].axvline(r_med_real, color='red')

ax[1].hist(g_arr_fake, bins=50, alpha=0.5, label='Fake')
ax[1].axvline(g_med_fake, color='green', label='Median')

ax[1].hist(g_arr_real, bins=50, alpha=0.5, label='Real')
ax[1].axvline(g_med_real, color='green')

ax[2].hist(b_arr_fake, bins=50, alpha=0.5, label='Fake')
ax[2].axvline(b_med_fake, color='blue', label='Median')

ax[2].hist(b_arr_real, bins=50, alpha=0.5, label='Real')
ax[2].axvline(b_med_real, color='blue')

ax[0].set_title('Red Channel')
ax[1].set_title('Green Channel')
ax[2].set_title('Blue Channel')

ax[0].set_xlabel('Pixel Value')
ax[1].set_xlabel('Pixel Value')
ax[2].set_xlabel('Pixel Value')

ax[0].set_ylabel('Count')
ax[1].set_ylabel('Count')
ax[2].set_ylabel('Count')

ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()

# save the figure
fig.savefig('miscl_median_pixel_values.png', dpi=300, bbox_inches='tight')

# compare the median  across the fake images and real images using t-tests
# import the ttest_ind function from the scipy.stats module
from scipy.stats import ttest_ind

# perform a t-test for each channel
r_ttest = ttest_ind(r_arr_fake, r_arr_real)
g_ttest = ttest_ind(g_arr_fake, g_arr_real)
b_ttest = ttest_ind(b_arr_fake, b_arr_real)

# print the p-values for each channel
print(f"p-value for red channel: {r_ttest.pvalue}")
print(f"p-value for green channel: {g_ttest.pvalue}")
print(f"p-value for blue channel: {b_ttest.pvalue}")


from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt

# create a list of all images
all_img_lst = real_img_lst + fake_img_lst

# Convert each PIL image to a NumPy array
all_img_lst = [np.array(img) for img in all_img_lst]

# Check the shape of each image in the list
for i, img in enumerate(all_img_lst):
    print(f"Image {i}: {img.shape}")

# process the images to be used in the clustering algorithm
# reshape each image to a 1D array
all_img_lst = [img.reshape(-1, 3) for img in all_img_lst]

# Check the shape of each image in the list
for i, img in enumerate(all_img_lst):
    print(f"Image {i}: {img.shape}")




# create a SpectralClustering object
sc = SpectralClustering(n_clusters=2, random_state=0)

# fit the clustering algorithm to the images
sc.fit(all_img_lst)

# get the cluster labels for each image
sc_labels = sc.labels_

#plot the cluster labels for each image
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
ax[0].scatter(range(len(all_img_labels)), all_img_labels, c=all_img_labels)
ax[0].set_title('True Labels')
ax[0].set_xlabel('Image Index')
ax[0].set_ylabel('Label')

ax[1].scatter(range(len(sc_labels)), sc_labels, c=sc_labels)
ax[1].set_title('Cluster Labels')
ax[1].set_xlabel('Image Index')
ax[1].set_ylabel('Label')

plt.show()
