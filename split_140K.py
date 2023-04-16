"""This is to split the data into train, test and validation sets."""

import os
import shutil
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# list all files in a directory

fakes = os.listdir("T9-140KRGB/Fake")
reals = os.listdir("T9-140KRGB/Real")
all_paths = fakes + reals

mapping_dict = {}
map_ls = []
label_ls = []

for i, file in enumerate(all_paths):
    if "Fake" in file:
        label = 0
    else:
        label = 1
    mapping_dict[i] = file
    map_ls.append(i)
    label_ls.append(label)


map_ls_arr = np.array(map_ls)
label_ls_arr = np.array(label_ls)

train_keys, test_keys = train_test_split(




# fake_ratio = len(fakes)/len(all_paths)
# real_ratio = len(reals)/len(all_paths)

# test_ratio = 0.2 * len(all_paths)
# validation_ratio = 0.1 * len(all_paths)
# train_ratio = 0.7 * len(all_paths)





for files in fakes:
    if random.random() < 0.2:
        shutil.move("T9-140KRGB/Fake/" + files, "T9-140KRGB/Test/Fake")
    elif random.random() < 0.2:
        shutil.move("T9-140KRGB/Fake/" + files, "T9-140KRGB/Validation/Fake")
    else:
        shutil.move("T9-140KRGB/Fake/" + files, "T9-140KRGB/Train/Fake")


# list all folders in a directory
# for folder in os.listdir(".\TemporaryFiles\Real"):
#     print(folder)
