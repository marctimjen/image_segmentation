# This file is used for taking the data from the dataset and shuffle and append it in a
# train and validation set.

import os
import numpy as np
import shutil

data_path = rf"/datasets/tumor_segmentation_pics_from_repo"

split = 0.13  # how many pct. for val.

np.random.seed(42)
print("Remember to change seed if necessary")

files = os.listdir(data_path)

start_path = "image_data"
if os.path.exists(start_path):
    shutil.rmtree(start_path)
os.makedirs(start_path)


train_path = "image_data/train"
os.makedirs(train_path)

train_path = "image_data/train/images"
os.makedirs(train_path)

train_path = "image_data/train/masks"
os.makedirs(train_path)


val_path = "image_data/val"
os.makedirs(val_path)

val_path = "image_data/val/images"
os.makedirs(val_path)

val_path = "image_data/val/masks"
os.makedirs(val_path)


control_list = [f for f in files if "control" in f]
control_val = np.random.choice(control_list, size=int(split * len(control_list)), replace=False)

val_path = "image_data/val"
train_path = "image_data/train"
for file in control_list:
    source_path_img = data_path + "/" + file
    source_path_mask = data_path + "/target_seg_" + file[-7:]
    if file in control_val:
        shutil.copy(source_path_img, val_path + "/images")
        shutil.copy(source_path_mask, val_path + "/masks")
    else:
        shutil.copy(source_path_img, train_path + "/images")
        shutil.copy(source_path_mask, train_path + "/masks")


patient_list = [f for f in files if "patient" in f]
patient_val = np.random.choice(patient_list, size=int(split * len(patient_list)), replace=False)

for file in patient_list:
    source_path_img = data_path + "/" + file
    source_path_mask = data_path + "/segmentation_" + file[-7:]
    if file in patient_val:
        shutil.copy(source_path_img, val_path + "/images")
        shutil.copy(source_path_mask, val_path + "/masks")
    else:
        shutil.copy(source_path_img, train_path + "/images")
        shutil.copy(source_path_mask, train_path + "/masks")
