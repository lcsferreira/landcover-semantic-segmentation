# separate masks from images in dataV2/train folder
# and save them in dataV2/masks folder
# and save images in dataV2/images folder

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# create folders for images and masks
Path("data/images").mkdir(parents=True, exist_ok=True)
Path("data/masks").mkdir(parents=True, exist_ok=True)

# get all files in data/train folder
files = os.listdir("data/train")

# separate masks from images (masks have _mask in their name)
for file in tqdm(files):
    if "_mask" in file:
        # read mask image
        mask = cv2.imread(f"data/train/{file}", cv2.IMREAD_UNCHANGED)
        # save mask image
        cv2.imwrite(f"data/masks/{file}", mask)
    else:
        # read image
        image = cv2.imread(f"data/train/{file}")
        # save image
        cv2.imwrite(f"data/images/{file}", image)

# check if the number of images and masks are the same
images = os.listdir("data/images")
masks = os.listdir("data/masks")
print(f"Number of images: {len(images)}")
print(f"Number of masks: {len(masks)}")
print(f"Images and masks are the same: {len(images) == len(masks)}")

