"""
Run this is in the main folder - yolo
Script used to split the data provided for Logo Detection project
by logo so that we can sample only the desired logos
"""


import os
import re
import shutil

import pandas as pd

# Specify all relevant paths
raw_data_path = os.path.join(os.getcwd(), "datasets", "DLCV_logo_project")
train_data_path = os.path.join(raw_data_path, "train")
split_data_path = os.path.join(raw_data_path, "train_by_logo")

try:
    train_images = os.listdir(train_data_path)

    # Create a directory for the split data by logo
    try:
        os.mkdir(split_data_path)
    except:
        print("Split data dir already exists!")

    # Extract all logos
    annotations = pd.read_csv(os.path.join(raw_data_path, "annot_train.csv"))
    logos = annotations["class"].unique()

    # Following two loops create the logo dirctories and move the images
    for logo in logos:
        try:
            os.mkdir(os.path.join(split_data_path, logo))
        except:
            print("{0} alredy exits".format(logo))

    print(train_data_path)
    for img in train_images:
        class_name = annotations[annotations["photo_filename"] == img][
            "class"
        ].unique()[0]
        shutil.move(
            os.path.join(train_data_path, img),
            os.path.join(split_data_path, class_name),
        )

    # Delete old empty train directory
    os.rmdir(train_data_path)

except:
    print("train folder doesn't exist, data might be already split")

# Rename the noise classes to NULL
annotations_noise = pd.read_csv(os.path.join(raw_data_path, "annot_noise.csv"))
annotations_noise["class"] = "NULL"
annotations_noise.to_csv(os.path.join(raw_data_path, "annot_noise.csv"), index=False)
