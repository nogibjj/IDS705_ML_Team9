"""Script to extract images of zip files into final folders"""

import os
import zipfile
import shutil
import glob
import numpy as np
import argparse
import tarfile


def extract_images(zip_file, output_dir):
    """Extract images from zip file into output directory"""
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)


# extract images from tar file
def extract_images_tar(tar_file, output_dir):
    """Extract images from tar file into output directory"""
    with tarfile.open(tar_file, "r") as tar_ref:
        tar_ref.extractall(output_dir)


### Retrieving Real Faces ###

# my code that got me the real images
n = 1000  # Have to tweak this according to the naming scheme of the zip files

for i in range(0, n):
    extract_images(
        r"C:\Users\ericr\Downloads\RealFaces0000" + str(i) + ".zip",
        r"C:\Users\ericr\Desktop\IDS 705 - Machine Learning\projectv2repo\IDS705_ML_Team9\TemporaryFiles\Real",
    )

# Reference for the above code
# extract_images(
#     r"C:\Users\ericr\Downloads\RealFaces00000.zip",
#     r"C:\Users\ericr\Desktop\IDS 705 - Machine Learning\projectv2repo\IDS705_ML_Team9\TemporaryFiles\Real",
# )

### Retrieving Real Faces ###


### Retrieving Fake Faces ###
# my code that got me the fake images
m = 1000  # Have to tweak this according to the naming scheme of the tar files
for i in range(0, m):
    extract_images_tar(
        r"C:\Users\ericr\Downloads\1m_faces_0" + str(i) + ".tar",
        r"C:\Users\ericr\Desktop\IDS 705 - Machine Learning\projectv2repo\IDS705_ML_Team9\TemporaryFiles\Fake",
    )

# Reference for the above code
# extract_images_tar(
#     r"C:\Users\ericr\Downloads\1m_faces_00.tar",
#     r"C:\Users\ericr\Desktop\IDS 705 - Machine Learning\projectv2repo\IDS705_ML_Team9\TemporaryFiles\Fake",
# )

### Retrieving Fake Faces ###


# Establishing the sample size
ratio = 0.1
np.random.seed(123)
real_samples = np.random.choice(n, n * ratio, replace=False)
fake_samples = np.random.choice(m, m * ratio, replace=False)


### Resizing and Erasing Images ###
### Image Extraction into the Sample and Final Datasets ###
from PIL import Image


def resize_and_erase_images(
    input_dir,
    output_dir,
    sample_dir,
    size,
    file_format,
    sample,
    flag=True,
    cap=1000,
):
    """Resize images in input directory and save to output directory"""
    file_format_string = "*." + file_format

    if flag is True:
        flag = "Real"
    else:
        flag = "Fake"

    iter = 1
    for image in glob.glob(os.path.join(input_dir, file_format_string)):
        img = Image.open(image)
        try:
            img = img.resize(size)

        except:
            print(f"Image at {iter+1} is not a valid image")
            continue

        img.save(os.path.join(output_dir, (flag + str(iter) + ".png")))
        if (iter - 1) in sample:
            img.save(os.path.join(sample_dir, (flag + str(iter) + ".png")))
        iter += 1
        # erase image
        os.remove(image)
        if iter >= (cap + 1):
            break
    print("Done with " + flag)


### Resizing and Erasing Images ###
### Image Extraction into the Sample and Final Datasets ###


# test run for hyperparameter sample images
size = (512, 512)  # Will downscale in model file
# has a scalability advantage if the directory structure does not vary, can be for looped
resize_and_erase_images(
    r"TempFiles/Real",
    "FullDataset/Real",
    "SampleDataset/Real",
    size,
    "png",
    flag=True,
    cap=1000,
)
resize_and_erase_images(
    r"TempFiles/Fake",
    "FullDataset/Fake",
    "SampleDataset/Fake",
    size,
    "jpg",
    flag=False,
    cap=1000,
)
