# importing cv2 and matplotlid
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# loading image
img = cv2.imread(
    r"C:\Users\ericr\Downloads\T9-Misc140KRGB-20230419T152541Z-001\T9-Misc140KRGB\trial.png"
)  # loading image
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# showing image using plt
# plt.imshow(img)
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(color_img)

prediction = DeepFace.analyze(color_img)


race_dictionary_real = {}
gender_dictionary_real = {}
bad_count_real = 0
# Looping through the faces in the folders

for image in os.listdir(
    r"C:\Users\ericr\Downloads\T9-Misc140KRGB-20230419T152541Z-001\T9-Misc140KRGB\Real"
):
    # print(image)
    path = r"C:\Users\ericr\Downloads\T9-Misc140KRGB-20230419T152541Z-001\T9-Misc140KRGB\Real"
    # join the path and filename
    file_path = os.path.join(path, image)
    img = cv2.imread(file_path)  # loading image

    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        prediction = DeepFace.analyze(color_img, enforce_detection=False)
    except:
        print("Error with: " + image)
        bad_count_real += 1
        continue

    race = prediction[0]["dominant_race"]
    gender = prediction[0]["dominant_gender"]

    if race not in race_dictionary_real:
        race_dictionary_real[race] = 1
    else:
        race_dictionary_real[race] += 1

    if gender not in gender_dictionary_real:
        gender_dictionary_real[gender] = 1

    else:
        gender_dictionary_real[gender] += 1


race_dictionary_fake = {}
gender_dictionary_fake = {}
bad_count_fake = 0

for image in os.listdir(
    r"C:\Users\ericr\Downloads\T9-Misc140KRGB-20230419T152541Z-001\T9-Misc140KRGB\Fake"
):
    # print(image)
    path = r"C:\Users\ericr\Downloads\T9-Misc140KRGB-20230419T152541Z-001\T9-Misc140KRGB\Fake"
    # join the path and filename
    file_path = os.path.join(path, image)
    img = cv2.imread(file_path)  # loading image

    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        prediction = DeepFace.analyze(color_img, enforce_detection=False)
    except:
        print("Error with: " + image)
        bad_count_fake += 1
        continue

    race = prediction[0]["dominant_race"]
    gender = prediction[0]["dominant_gender"]

    if race not in race_dictionary_fake:
        race_dictionary_fake[race] = 1
    else:
        race_dictionary_fake[race] += 1

    if gender not in gender_dictionary_fake:
        gender_dictionary_fake[gender] = 1

    else:
        gender_dictionary_fake[gender] += 1
