import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# !pip install -q -U segmentation-models-pytorch albumentations > /dev/null
# !pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp

from helpers import colour_code_segmentation, reverse_one_hot

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    # Optional argument for image format
    parser.add_argument('--jpg', dest='image_format', action='store_const', const='jpg', help='Set image format to jpg')
    parser.add_argument('--png', dest='image_format', action='store_const', const='png', help='Set image format to png')

    # Optional arguments with specific values
    parser.add_argument('--data-dir', type=str, help='Directory containing the data')
    parser.add_argument('--encoder', type=str, help='Encoder to use, e.g., resnet50')
    parser.add_argument('--weights', type=str, help='Weights to use, e.g., imagenet')
    parser.add_argument('--input-folder', type=str, help='Input folder containing images')
    parser.add_argument('--output-folder', type=str, help='Output folder to save images')
    parser.add_argument('--model-path', type=str, help='Path to the model')
    parser.add_argument('--image-size', type=int, help='Image size')

    args = parser.parse_args()
    return args

args = parse_args()
image_format = args.image_format
data_dir = args.data_dir
encoder = args.encoder
encoder_weights = args.weights
image_size = args.image_size
input_folder = args.input_folder
output_folder = args.output_folder
model_path = args.model_path

if image_size < 256 or image_size > 256:
    print("Image size should be 256")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
if os.path.exists(model_path):
    model = torch.load(model_path, map_location=DEVICE)
    print("Model loaded successfully")
else:
    print("Model not found")
    sys.exit(1)
    
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("Output folder created")

#get all imagges
images_list = [os.path.join(input_folder, img) for img in os.listdir(input_folder) if img.endswith('.{}'.format(image_format))]

ENCODER = encoder
ENCODER_WEIGHTS = encoder_weights
DATA_DIR = data_dir

class_dict = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()

# Useful to shortlist specific classes in datasets with large number of classes
select_classes = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

for idx, tile_img in enumerate(images_list):
  print(f"Processing image: {tile_img}")
  image = np.array(Image.open(tile_img))
  image = preprocessing_fn(image)
  x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0).float()
  x_tensor = x_tensor.permute(0,3,1,2)
  # Predict test image
  pred_mask = model(x_tensor)
  pred_mask = pred_mask.detach().squeeze().cpu().numpy()
  # Convert pred_mask from `CHW` format to `HWC` format
  pred_mask = np.transpose(pred_mask,(1,2,0))
  # Get prediction channel corresponding to foreground
  pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
  cv2.imwrite(os.path.join(output_folder, f"pred_{idx}.png"), np.hstack([pred_mask])[:,:,::-1])