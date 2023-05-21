# importing cv2 and matplotlid\b
import cv2
import tensorflow as tflow
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
import json
import gc
# Apparently for monitoring, it only shows in terminals
# tflow.debugging.set_log_device_placement(True)

# Set up Prefetching
AUTOTUNE = tflow.data.AUTOTUNE

print("Num GPUs Available: ", len(tflow.config.list_physical_devices("GPU")))
print(tflow.config.list_physical_devices("GPU"))
print("Num CPUs Available: ", len(tflow.config.list_physical_devices("CPU")))
print(tflow.config.list_physical_devices("CPU"))




# loading image
img = cv2.imread(
    r"C:\Users\Eric-DQGM\Downloads\MLProject\IDS705_ML_Team9\T9-Val\Fake\Fake44.png"
)  
# loading image
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# showing image using plt
plt.imshow(img)
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(color_img)

prediction = DeepFace.analyze(color_img)


# Looping through the faces in the folders

pathing_all_sets = ['T9-Val', 'T9-Test', 'T9-Train']
subset_dn = dict()
bad_files = dict()
categories = ["Real", "Fake"]

checkpoint = True
cp_counter = 0

if checkpoint:
    dns_backup = json.load(open('./checkpoint_dn.json', 'r'))
    if dns_backup["last folder"] != 'T9-Val':
        subset_dn = json.load(open('./race_gender_classifier.json', 'r'))
        bad_files = json.load(open('./bad_files.json', 'r'))
    pathing_all_sets = pathing_all_sets[(pathing_all_sets.index((dns_backup["last folder"]))):]

for i in range(0,100):
    
    print((("Starting from: " + pathing_all_sets[0]) + "\n"))

for folder in pathing_all_sets: # All sets
    
    race_subset = dict({"Real" : dict(), "Fake" : dict()})
    gender_subset = dict({"Real" : dict(), "Fake" : dict()})
    bad_images = dict({"Real" : [], "Fake" : []})
    bad_count_dn = dict({"Real Count" : 0, "Fake Count" : 0})
    
    if checkpoint and folder == dns_backup["last folder"]:
        race_subset = dns_backup["race subset"].copy()
        gender_subset = dns_backup["gender subset"].copy()
        bad_images = dns_backup["bad images"].copy()
        bad_count_dn = dns_backup["bad count"].copy()

    for category in categories: # Real and Fake

        if checkpoint and folder == dns_backup["last folder"]:

            category = dns_backup["last category"]
        
        new_path = os.path.join(r"C:\Users\Eric-DQGM\Downloads\MLProject\IDS705_ML_Team9", folder, category)
        
        all_images = os.listdir(new_path)

        if checkpoint and folder == dns_backup["last folder"]:

            all_images = all_images[(all_images.index(dns_backup["last image"])):]

            checkpoint = False

        length_folder = len(all_images)

        for i in range(0, len(all_images)):
            image = all_images[i]
            file_path = os.path.join(new_path, image)
               
            img = cv2.imread(file_path)  # loading image
            
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            try:
                
                prediction = DeepFace.analyze(color_img, enforce_detection=True)
            
            except:
                
                try:
                        
                    prediction = DeepFace.analyze(color_img, enforce_detection=False)
                        
                
                except:
                    
                    print("Error with: " + image)
                    bad_images[category].extend([image])
                    bad_count_dn[category+" Count"] += 1
                    continue

            race = prediction[0]["dominant_race"]
            gender = prediction[0]["dominant_gender"]

            if race not in race_subset[category]:
                 
                race_subset[category][race] = 1

            else:
                race_subset[category][race] += 1


            if gender not in gender_subset[category]:

                gender_subset[category][gender] = 1

            else:
                gender_subset[category][gender] += 1

            cp_counter += 1

            if cp_counter == 500:
                                
                dns_to_write = {"race subset" : race_subset, 
                                "gender subset": gender_subset, 
                                "bad images" : bad_images, 
                                "bad count" :bad_count_dn,
                                "last image" : image,
                                "last folder" : folder,
                                "last category" : category}

                with open(f'checkpoint_dn.json', 'w') as fp:
                    json.dump(dns_to_write, fp)

                cp_counter = 0

                for i in range(0,10):
                    
                    print(f"Checkpoint reached at {folder}/{category}, {(i+1)/length_folder:.2f} % completed")

    subset_dn[folder] = dict({"Race" : race_subset, "Gender" : gender_subset})
    bad_files[folder] = dict({"Bad Images" : bad_images, "Bad Count" : bad_count_dn})

    with open('race_gender_classifier.json', 'w') as fp:
        json.dump(subset_dn, fp)
    with open('bad_files.json', 'w') as fp:
        json.dump(bad_files, fp)

    for i in range(0,100):
        print(f"Completed {folder} \n")

# writing the final results

with open('final_race_gender_classifier.json', 'w') as fp:
    json.dump(subset_dn, fp)
with open('final_bad_files.json', 'w') as fp:
    json.dump(bad_files, fp)

# Tally the total results across all data 

race_tally = dict({"Real" : None, "Fake" : None})
gender_tally = dict({"Real" : None, "Fake" : None})
subset_dn = json.load(open('./final_race_gender_classifier.json', 'r'))
bad_files = json.load(open('./final_bad_files.json', 'r'))

for folder in subset_dn:

    folder_subset = subset_dn[folder]

    race_dn = folder_subset["Race"]
    
    for category in race_dn: # Real and Fake

        if race_tally[category] is None:

            race_tally[category] = race_dn[category].copy()

        else:

            for key in race_dn[category]:

                if key in race_tally[category]:

                    race_tally[category][key] += race_dn[category][key]
            
                else:

                    race_tally[category][key] = race_dn[category][key]

    gender_dn = folder_subset["Gender"]
    
    for category in gender_dn: # Real and Fake

        if gender_tally[category] is None:

            gender_tally[category] = gender_dn[category].copy()

        else:

            for key in gender_dn[category]:

                if key in gender_tally[category]:

                    gender_tally[category][key] += gender_dn[category][key]
            
                else:

                    gender_tally[category][key] = gender_dn[category][key]


# writing the final tallies

with open('final_tally_race.json', 'w') as fp:
    json.dump(race_tally, fp)
with open('final_tally_gender.json', 'w') as fp:
    json.dump(gender_tally, fp)






# for image in os.listdir(r"C:\Users\Eric-DQGM\Downloads\MLProject\IDS705_ML_Team9\T9-Val\Real"):  
#         c += 1
#         path = r"C:\Users\Eric-DQGM\Downloads\MLProject\IDS705_ML_Team9\T9-Val\Real"
#         # join the path and filename
#         file_path = os.path.join(path, image)
#         img = cv2.imread(file_path)  # loading image
#         color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         try:
#             prediction = DeepFace.analyze(color_img, enforce_detection=True)
            
#         except:
#              try:
#                   prediction = DeepFace.analyze(color_img, enforce_detection=False)
                  
#              except:
#                 print("Error with: " + image)
#                 bad_count_real += 1
#                 continue
#         race = prediction[0]["dominant_race"]
#         gender = prediction[0]["dominant_gender"]

# #         if race not in race_dictionary_real:
# #             race_dictionary_real[race] = 1
# #         else:
# #             race_dictionary_real[race] += 1

# #         if gender not in gender_dictionary_real:
# #             gender_dictionary_real[gender] = 1

# #         else:
# #             gender_dictionary_real[gender] += 1

# #     # write gender_dictionary_real to a txt file

# #     with open("gender_race_dn_real.txt", "w") as f:

# #         f.write(f"{race_dictionary_real}")
# #         f.write(f"{gender_dictionary_real}")
# #         f.write(f"{bad_count_real}")


# #     race_dictionary_fake = {}
# #     gender_dictionary_fake = {}
# #     bad_count_fake = 0

# #     for image in os.listdir(
# #         r"C:\Users\Eric-DQGM\Downloads\MLProject\IDS705_ML_Team9\T9-140KRGB\Fake"
# #     ):
# #         print(image)
# #         path = r"C:\Users\Eric-DQGM\Downloads\MLProject\IDS705_ML_Team9\T9-140KRGB\Fake"
# #         # join the path and filename
# #         file_path = os.path.join(path, image)
# #         img = cv2.imread(file_path)  # loading image

# #         color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# #         try:
# #             prediction = DeepFace.analyze(color_img, enforce_detection=False)
# #         except:
# #             print("Error with: " + image)
# #             bad_count_fake += 1
# #             continue

# #         race = prediction[0]["dominant_race"]
# #         gender = prediction[0]["dominant_gender"]

# #         if race not in race_dictionary_fake:
# #             race_dictionary_fake[race] = 1
# #         else:
# #             race_dictionary_fake[race] += 1

# #         if gender not in gender_dictionary_fake:
# #             gender_dictionary_fake[gender] = 1

# #         else:
# #             gender_dictionary_fake[gender] += 1

# #     # write results to a txt file


# #     with open("gender_race_dn_fake.txt", "w") as f:

# #         f.write(f"{race_dictionary_fake}")
# #         f.write(f"{gender_dictionary_fake}")
# #         f.write(f"{bad_count_fake}")