"""Script to extract images of zip files into final folders"""

import os
import zipfile
import shutil
import glob

# import cv2
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


# my code that got me the real images
extract_images(
    r"C:\Users\ericr\Downloads\RealFaces00000.zip",
    ".\TemporaryFiles\Real",
)

# my code that got me the fake images
extract_images_tar(
    r"C:\Users\ericr\Downloads\1m_faces_00.tar",
    ".\TemporaryFiles\Fake",
)

# def resize_images(input_dir, output_dir, size):
#     """Resize images in input directory and save to output directory"""
#     for image in glob.glob(os.path.join(input_dir, '*.jpg')):
#         img = cv2.imread(image)
#         img = cv2.resize(img, size)
#         cv2.imwrite(os.path.join(output_dir, os.path.basename(image)), img)

# erase image files in output directory
def erase_images(output_dir, format):
    """Erase images in output directory"""
    for image in glob.glob(os.path.join(output_dir, "*." + format)):
        os.remove(image)


# for directory in C:\Users\ericr\Downloads\TargetErase\Fake
# save the images to C:\Users\ericr\Downloads\Team9images\Fake
# for directory in C:\Users\ericr\Downloads\TargetErase\Real
# save the images to C:\Users\ericr\Downloads\Team9images\Real

from PIL import Image


def resize_and_erase_images(
    input_dir, output_dir, size, file_format, flag=True, cap=1000
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
            # grayscale
            img = img.convert("L")

        except:
            print(f"Image at {iter+1} is not a valid image")
            continue

        img.save(os.path.join(output_dir, (flag + str(iter) + ".png")))
        iter += 1
        # erase image
        os.remove(image)
        if iter >= (cap + 1):
            break
    print("Done with " + flag)


# test run for real images
size = (256, 256)
# has a scalability advantage if the directory structure does not vary, can be for looped
resize_and_erase_images(
    ".\TemporaryFiles\Real\\00000",
    ".\grayscale1kdraft\Real",
    size,
    "png",
    flag=True,
)
resize_and_erase_images(
    ".\TemporaryFiles\Fake\\1m_faces_00",
    ".\grayscale1kdraft\Fake",
    size,
    "jpg",
    flag=False,
)

# cleaning the 1m_faces_00 directory
erase_images(".\TemporaryFiles\Fake\\1m_faces_00", "jpg")


# def main():
#     """Main function"""
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--input_dir",
#         type=str,
#         default="C:\\Users\\ericr\\Downloads\\TargetErase\\Fake",
#         help="Input directory",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="C:\\Users\\ericr\\Downloads\\Team9images\\Fake",
#         help="Output directory",
#     )
#     parser.add_argument(
#         "--size", type=int, nargs=2, default=(64, 64), help="Output image size"
#     )
#     args = parser.parse_args()

#     # erase_images(args.output_dir)
#     # extract_images(args.input_dir, args.output_dir)
#     # extract_images_tar(args.input_dir, args.output_dir)
#     resize_images(args.input_dir, args.output_dir, args.size)
