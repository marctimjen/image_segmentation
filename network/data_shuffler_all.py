# This file is used for taking the data from the dataset and shuffle and append it in a
# train and validation set.

import os
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser(description='data-shuffler')
parser.add_argument("-t", "--all", required=True, help="what data to shuffle")
args = parser.parse_args()

data_path = rf"/datasets/tumor_segmentation_pics_from_repo"

split = 0.15  # how many pct. for val.

np.random.seed(13)  # 42
print("Remember to change seed if necessary")

files = os.listdir(data_path)

all_data = True

if args.t == "a":
    print("use all data")
    start_path = f"image_data_all2"
    data_path = rf"/datasets/tumor_segmentation_mnm_anno_v2"
elif args.t == "n":
    print("use only repo data")
    all_data = False
    start_path = f"image_data"
else:
    print("use data without our annotations")
    start_path = f"image_data_all"
    data_path = rf"/datasets/tumor_segmentation_mnm_anno"

if os.path.exists(start_path):
    shutil.rmtree(start_path)
os.makedirs(start_path)


train_path = f"{start_path}/train"
os.makedirs(train_path)

train_path = f"{start_path}/train/images"
os.makedirs(train_path)

train_path = f"{start_path}/train/masks"
os.makedirs(train_path)


val_path = f"{start_path}/val"
os.makedirs(val_path)

val_path = f"{start_path}/val/images"
os.makedirs(val_path)

val_path = f"{start_path}/val/masks"
os.makedirs(val_path)


control_list = [f for f in files if "control" in f]
control_val = np.random.choice(control_list, size=int(split * len(control_list)), replace=False)

val_path = f"{start_path}/val"
train_path = f"{start_path}/train"
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


# after the split of their data, we can move the rest of the files into training:

if all_data:
    rest_files = os.listdir(data_path)

    patient_list = [f for f in rest_files if "patient" in f]
    for file in patient_list:
        source_path_img = data_path + "/" + file
        source_path_mask = data_path + "/segmentation_" + file[-7:]
        shutil.copy(source_path_img, train_path + "/images")
        shutil.copy(source_path_mask, train_path + "/masks")


    control_list = [f for f in rest_files if "control" in f]
    for file in control_list:
        source_path_img = data_path + "/" + file
        source_path_mask = data_path + "/target_seg_" + file[-7:]
        shutil.copy(source_path_img, train_path + "/images")
        shutil.copy(source_path_mask, train_path + "/masks")

