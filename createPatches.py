import os
import cv2
import numpy as np
from PIL import Image
from patchify import patchify
from pathlib import Path
from collections import namedtuple

root_dir = 'data/'
patch_size = 256
Path("data/patches").mkdir(parents=True, exist_ok=True)

Label = namedtuple('Label', ['name', 'color'])
labels = [
    Label('urban_land', (0, 255, 255)),
    Label('agriculture_land', (255, 255, 0)),
    Label('rangeland', (255, 0, 255)),
    Label('forest_land', (0, 255, 0)),
    Label('water', (0, 0, 255)),
    Label('barren_land', (255, 255, 255)),
    Label('unknown', (0, 0, 0))
]

color2label = {label.color: label for label in labels}

image_dir= root_dir + 'images'
for path, subdirs, files in os.walk(image_dir):
    #print(path)  
    dirname = path.split(os.path.sep)[-1]
    #print(dirname)
    images = os.listdir(path)  #List of all image names in this subdirectory
    #print(images)
    for i, image_name in enumerate(images):  
        if image_name.endswith(".jpg"):
            #print(image_name)
            image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
            SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
            SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
            image = Image.fromarray(image)
            image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
            #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
            image = np.array(image)             

            #Extract patches from each image
            print("Now patchifying image:", path+"/"+image_name)
            #remove .png from image_name
            image_name = image_name[:-4]
            patches_img = patchify(image, (256, 256, 3), step=256)  #Step=256 for 256 patches means no overlap
    
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    
                    single_patch_img = patches_img[i,j,:,:]
                    #single_patch_img = (single_patch_img.astype('float32')) / 255. #We will preprocess using one of the backbones
                    single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                    cv2.imwrite(root_dir+"patches/"+
                               image_name+"_"+str(i)+"_"+str(j)+".jpg", single_patch_img)
                    # image_dataset.append(single_patch_img)
                    
#                      #Now do the same as above for masks
#  #For this specific dataset we could have added masks to the above code as masks have extension png
mask_dir=root_dir+"masks"
for path, subdirs, files in os.walk(mask_dir):
    #print(path)  
    dirname = path.split(os.path.sep)[-1]

    masks = os.listdir(path)  #List of all image names in this subdirectory
    for i, mask_name in enumerate(masks):  
        if mask_name.endswith(".png"):           
            mask = cv2.imread(path+"/"+mask_name, 1)  #Read each image as BGR 
            
            SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
            SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
            mask = Image.fromarray(mask)
            mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
            #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
            mask = np.array(mask)     
   
            #Extract patches from each image
            print("Now patchifying mask:", path+"/"+mask_name)
            
            mask_name = mask_name[:-4]
            patches_mask = patchify(mask, (256, 256, 3), step=256)  #Step=256 for 256 patches means no overlap
    
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    
                    single_patch_mask = patches_mask[i,j,:,:]
                    #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                    single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.  
                                                
                    cv2.imwrite(root_dir+"patches/"+
                               mask_name+"_"+str(i)+"_"+str(j)+".png", single_patch_mask)
                    
import pandas as pd
import os

# Function to create the DataFrame
def create_dataframe(images_dir, masks_dir):
    # List all image files and mask files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

    # Ensure there is a corresponding mask for each image
    assert len(image_files) == len(mask_files), "Number of images and masks must be equal"

    # Create the DataFrame
    data = []
    for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        image_id = i + 1
        sat_image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)
        data.append([image_id, sat_image_path, mask_path])
    
    df = pd.DataFrame(data, columns=["image_id", "sat_image_path", "mask_path"])
    return df

# Example usage
images_dir = 'data/patches'
masks_dir = 'data/patches'
df = create_dataframe(images_dir, masks_dir)
df.to_csv('data/metadata_patches.csv', index=False)
