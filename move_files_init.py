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
Path("dataDeepGlobe/images").mkdir(parents=True, exist_ok=True)
Path("dataDeepGlobe/masks").mkdir(parents=True, exist_ok=True)

# get all files in dataDeepGlobe/train folder
files = os.listdir("dataDeepGlobe/train")

# separate masks from images (masks have _mask in their name)
for file in tqdm(files):
    if "_mask" in file:
        # read mask image
        mask = cv2.imread(f"dataDeepGlobe/train/{file}", cv2.IMREAD_UNCHANGED)
        # save mask image
        cv2.imwrite(f"dataDeepGlobe/masks/{file}", mask)
    else:
        # read image
        image = cv2.imread(f"dataDeepGlobe/train/{file}")
        # save image
        cv2.imwrite(f"dataDeepGlobe/images/{file}", image)

# check if the number of images and masks are the same
images = os.listdir("dataDeepGlobe/images")
masks = os.listdir("dataDeepGlobe/masks")
print(f"Number of images: {len(images)}")
print(f"Number of masks: {len(masks)}")
print(f"Images and masks are the same: {len(images) == len(masks)}")

