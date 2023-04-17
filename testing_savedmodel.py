import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import matplotlib.pyplot as plt
# PIL
from PIL import Image
import glob

#model = ResNet50(weights=None)

# model.save('./demo_resnet_model_weights_140K.h5', save_format='tf')

model = tf.keras.models.load_model('./140k_model/deploy_resnet_model_140K_alldata_rgb.h5')


weights = model.load_weights('./140k_model/deploy_resnet_model_140K_alldata_rgb_weights.h5', by_name=True)

# Load the image data
image_path = './Data/Application/MIDS24/'
new_image_path = './Data/Application/resizedMIDS'


def pre_process_image_data(path):
    for i in os.listdir(path):
        if i.endswith(".jpg"):
            # load the image and resize to 256 by 256
            img = Image.open(path + i)
            # save it to the same folder and overwrite the original
            # resize to 256 x 256
            img = img.resize((256, 256))
            print(img.size)
            img.save(new_image_path + i)
        elif i.endswith(".png"):
            # load the image and resize to 256 by 256
            img = Image.open(path + i)
            # save it to the same folder and overwrite the original
            # resize to 256 x 256
            img = img.resize((256, 256))
            #print image size
            print(img.size)
            img.save(new_image_path + i)

        elif i.endswith(".jpeg"):
            # load the image and resize to 256 by 256
            img = Image.open(path + i)
            # save it to the same folder and overwrite the original
            # resize to 256 x 256
            img = img.resize((256, 256))
            print(img.size)
            img.save(new_image_path + i)


pre_process_image_data(image_path)







###############################




# read in the first 100 images
img_lst = []

for filename in glob.glob(image_path):
    with Image.open(filename) as img:
        # convert to jpg
        img = img.convert('RGB')
        # resize the image
        #img = img.resize((128, 128))
        # resize to 256 x 256
        img = img.resize((256, 256))
        img_lst.append(img)

# check the resolution of the images
for img in img_lst:
    print(f"Image resolution: {img.size}")


print(img_lst)

# convert to numpy array
img_arr = np.array(img_lst)
















img = tf.keras.preprocessing.image.load_img(
        r" ".format(image),
        target_size=(256, 256),
    )

# Set up input parameters
img_folder = './Data/Application/MIDS24/*.jpeg'
img_size = (256, 256)


# Loop through images in folder and make predictions
for img_file in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_file)
    img = Image.load_img(img_path, target_size=img_size)
    x = Image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    preds_decoded = decode_predictions(preds, top=3)[0]
    print('Predicted classes for image', img_file, ':')
    for pred in preds_decoded:
        print(pred[1], ':', pred[2])