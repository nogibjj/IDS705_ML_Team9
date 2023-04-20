import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import os

model_rgb_path = "./140k_model/deploy_resnet_model_140K_alldata_rgb.h5"
weights_rgb_path = "./140k_model/deploy_resnet_model_140K_alldata_rgb_weights.h5"

# load the model using h5py
import h5py
model_rgb = h5py.File(model_rgb_path, 'r')
model_rgb_weights = h5py.File(weights_rgb_path, 'r')

# load the model using keras
from keras.models import model_from_json
import tensorflow.keras as keras

# Load the h5 model file
model = keras.models.load_model(model_rgb_path)

# Print the model summary
model.summary()

#Loading the Images
from PIL import Image
def pre_process_image_data(path):
    lst_img = []
    for i in os.listdir(path):
        # print(i)
        if i[-3:] == "png":
        # print(path + str('/' + i))
            with open(path + str('/' + i), 'rb') as f:
                img = Image.open(f)
                # resize the image to 256 by 256
                img = img.resize((256, 256))
                # save it to the same folder and overwrite the original
                img.save(path + str('/' + i))
                # print("saved")
                # store their array in a list
                lst_img.append(np.array(img))
    return lst_img


# mids24 prediction
ai_path = "/workspaces/IDS705_ML_Team9/Data/Application/very_real_ai-faces"

# load the image data
pre_process_image_data(ai_path)

