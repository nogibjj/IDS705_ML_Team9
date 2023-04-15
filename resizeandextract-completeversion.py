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


# erase images before initiating image extraction
def erase_images(output_dir, format):
    """Erase images in output directory"""
    for image in glob.glob(os.path.join(output_dir, "*." + format)):
        os.remove(image)


### Cleaning Commences ###
# Cleaning out the directories
for folder in os.listdir(".\TemporaryFiles\Real"):
    try:
        os.remove(os.path.join(".\TemporaryFiles\Real", folder))
    except:
        print("Not a file")

        try:
            os.rmdir(os.path.join(".\TemporaryFiles\Real", folder))
        except:
            raise Exception("Not a folder")

for folder in os.listdir(".\TemporaryFiles\Fake"):
    try:
        os.remove(os.path.join(".\TemporaryFiles\Fake", folder))
    except:
        print("Not a file")
        try:
            os.rmdir(os.path.join(".\TemporaryFiles\Fake", folder))
        except:
            raise Exception("Not a folder")

### Cleaning Complete ###
### Retrieving Real Faces ###

# the code that got me the real images

real_face_path = r"C:\Users\Eric-DQGM\Downloads\MLProject\RealFaces"
fake_face_path = r"C:\Users\Eric-DQGM\Downloads\MLProject\FakeFaces"

### Real Faces Extraction ###

for real_facez_zip in glob.glob(os.path.join(real_face_path, "*.zip")):
    print(real_facez_zip)
    extract_images(
        real_facez_zip,
        ".\TemporaryFiles\Real",
    )


# list all folders in a directory
for folder in os.listdir(".\TemporaryFiles\Real"):
    for img in glob.glob(os.path.join(".\TemporaryFiles\Real", folder + "\*.png")):
        print(img)
        shutil.move(img, ".\TemporaryFiles\Real")
    # remove folder after moving images
    os.rmdir(os.path.join(".\TemporaryFiles\Real", folder))


### Real Faces Extraction Complete ###


### Fake Faces Extraction ###
for tar_file in glob.glob(os.path.join(fake_face_path, "*.tar")):
    print(tar_file)
    extract_images_tar(
        tar_file,
        ".\TemporaryFiles\Fake",
    )


# list all folders in a directory
for folder in os.listdir(".\TemporaryFiles\Fake"):
    print(folder, "first print")
    if not (os.path.isdir(os.path.join(".\TemporaryFiles\Fake", folder))):
        print("Not a directory")

        # if it was an image, don't touch it
        try:
            Image.open(os.path.join(".\TemporaryFiles\Fake", folder))
        except:
            print("Not an image")
            os.remove(os.path.join(".\TemporaryFiles\Fake", folder))

    else:  # if it was a directory, move all images to the main folder
        print("Is a directory")
        for img in glob.glob(os.path.join(".\TemporaryFiles\Fake", folder + "\*.jpg")):
            print(img)
            shutil.move(img, ".\TemporaryFiles\Fake")
        # remove folder after moving images
        # print(img, folder)
        os.rmdir(os.path.join(".\TemporaryFiles\Fake", folder))

### Fake Faces Extraction Complete ###


# length of real images and fake images
real_images = len(os.listdir(".\TemporaryFiles\Real"))
fake_images = len(os.listdir(".\TemporaryFiles\Fake"))
print("There are", real_images, "real images")
print("There are", fake_images, "fake images")

# choosing the cap for the image count
image_cap = min(real_images, fake_images)

### Resizing and Erasing Images ###
### Image Extraction into the Final Datasets ###
from PIL import Image


def resize_and_erase_images(
    input_dir: str,
    output_dir: str,
    gray_output_dir: str,
    size: tuple,
    file_format: str,
    cap: int = image_cap,
) -> int:
    """Resize images in input directory and save to output directory"""
    file_format_string = "*." + file_format

    if "Real" in input_dir:
        flag = "Real"
    else:
        flag = "Fake"

    iter = 1
    for image in glob.glob(os.path.join(input_dir, file_format_string)):
        img = Image.open(image)
        try:
            img = img.resize(size)
            img_gray = img.convert("L")

        except:
            print(f"Image at {iter} is not a valid image")
            os.remove(image)
            continue

        img.save(os.path.join(output_dir, (flag + str(iter) + ".png")))
        img_gray.save(os.path.join(gray_output_dir, (flag + str(iter) + "_gray.png")))
        iter += 1
        # erase image
        os.remove(image)
        if iter >= cap:
            break
        print(f"Image count is {iter-1} for {flag} images")
    print("Done with " + flag)
    return iter - 1


im_size = (256, 256)  # Will downscale in model file
# has a scalability advantage if the directory structure does not vary, can be for looped


cap_imposed_by_real = resize_and_erase_images(
    ".\TemporaryFiles\Real",
    "tenKDataset\Real",
    "tenKGrayDataset\Real",
    size=im_size,
    file_format="png",
    cap=image_cap,
)

print(cap_imposed_by_real)

resize_and_erase_images(
    ".\TemporaryFiles\Fake",
    "tenKDataset\Fake",
    "tenKGrayDataset\Fake",
    size=im_size,
    file_format="jpg",
    cap=cap_imposed_by_real,
)


# balance the dataset
real_num = len(os.listdir("tenKDataset\Real"))
fake_num = len(os.listdir("tenKDataset\Fake"))
print("Final Real Images:", real_num)
print("Final Fake Images:", fake_num)


### Balancing the Dataset : Code here if needed ###

# import numpy as np

# if real_num > fake_num:

#     for i in range(real_num - fake_num):
#         np.random.seed(i)
#         j = np.random.randint(0, real_num)
#         os.remove("tenKDataset\Real\Real" + str(j) + ".png")

# elif fake_num > real_num:
#     for i in range(fake_num - real_num):
#         np.random.seed(i)
#         j = np.random.randint(0, fake_num)
#         os.remove("tenKDataset\Fake\Fake" + str(j) + ".png")

# else: # n == m
#     print("The dataset is balanced")
#     pass


### Archiving the Sample Code for sampling images ###


# ratio = 0.1
# np.random.seed(123)
# real_samples = np.random.choice(n, n * ratio, replace=False)
# fake_samples = np.random.choice(m, m * ratio, replace=False)
### Archiving the Sample Code for sampling images ###


### Resizing and Erasing Images ###
### Image Extraction into the Final Datasets ###
# from PIL import Image


# def resize_and_erase_images(
#     input_dir,
#     output_dir,
#     sample_dir,
#     size,
#     file_format,
#     sample,
#     flag=True,
#     cap=1000,
# ):
#     """Resize images in input directory and save to output directory"""
#     file_format_string = "*." + file_format

#     if flag is True:
#         flag = "Real"
#     else:
#         flag = "Fake"

#     iter = 1
#     for image in glob.glob(os.path.join(input_dir, file_format_string)):
#         img = Image.open(image)
#         try:
#             img = img.resize(size)

#         except:
#             print(f"Image at {iter+1} is not a valid image")
#             continue

#         img.save(os.path.join(output_dir, (flag + str(iter) + ".png")))
#         if (iter - 1) in sample:
#             img.save(os.path.join(sample_dir, (flag + str(iter) + ".png")))
#         iter += 1
#         # erase image
#         os.remove(image)
#         if iter >= (cap + 1):
#             break
#     print("Done with " + flag)
