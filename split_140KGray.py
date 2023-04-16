"""This is to split the data into train, test and validation sets."""

import os
import shutil
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# list all files in a directory

fakes = os.listdir("T9-140KGray/Fake")
reals = os.listdir("T9-140KGray/Real")
all_paths = fakes + reals
labels_fake = [0] * len(fakes)
labels_real = [1] * len(reals)
all_labels = labels_fake + labels_real


train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=all_labels,
)

train_paths, validation_paths, train_labels, validation_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.1,
    random_state=42,
    shuffle=True,
    stratify=train_labels,
)

# copy the imafges to the new folders
for files in train_paths:
    if files in fakes:
        shutil.copy("T9-140KGray/Fake/" + files, "T9-GrayTrain/Fake")
    else:
        shutil.copy("T9-140KGray/Real/" + files, "T9-GrayTrain/Real")

for files in test_paths:
    if files in fakes:
        shutil.copy("T9-140KGray/Fake/" + files, "T9-GrayTest/Fake")
    else:
        shutil.copy("T9-140KGray/Real/" + files, "T9-GrayTest/Real")

for files in validation_paths:
    if files in fakes:
        shutil.copy("T9-140KGray/Fake/" + files, "T9-GrayVal/Fake")
    else:
        shutil.copy("T9-140KGray/Real/" + files, "T9-GrayVal/Real")

# fake_ratio = len(fakes)/len(all_paths)
# real_ratio = len(reals)/len(all_paths)

# test_ratio = 0.2 * len(all_paths)
# validation_ratio = 0.1 * len(all_paths)
# train_ratio = 0.7 * len(all_paths)


# list all folders in a directory
# for folder in os.listdir(".\TemporaryFiles\Real"):
#     print(folder)
