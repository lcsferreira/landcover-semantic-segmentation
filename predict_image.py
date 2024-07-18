import sys
import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt

# !pip install -q -U segmentation-models-pytorch albumentations > /dev/null
# !pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp

from helpers import colour_code_segmentation
from smooth_tiled_predictions import predict_img_with_smooth_windowing

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    # Optional arguments with specific values
    parser.add_argument('--data-dir', type=str, help='Directory containing the data')
    parser.add_argument('--input-image', type=str, help='Input image to predict')
    parser.add_argument('--output-image-name', type=str, help='Output image name')
    parser.add_argument('--encoder', type=str, help='Encoder to use, e.g., resnet50')
    parser.add_argument('--weights', type=str, help='Weights to use, e.g., imagenet')
    parser.add_argument('--model-path', type=str, help='Path to the model')
    parser.add_argument('--patch-size', type=int, help='Patch size')
    parser.add_argument('--n-divisions', type=int, help='Number of divisions to overlap') # Minimal amount of overlap for windowing. Must be an even number.
    parser.add_argument('--n-classes', type=int, help='Number of classes in the dataset')

    args = parser.parse_args()
    return args
  
args = parse_args()
input_image = args.input_image
output_image_name = args.output_image_name
ENCODER = args.encoder
ENCODER_WEIGHTS = args.weights
model_path = args.model_path
patch_size = args.patch_size
n_divisions = args.n_divisions
n_classes = args.n_classes
DATA_DIR = args.data_dir

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
if os.path.exists(model_path):
    model = torch.load(model_path, map_location=DEVICE)
    print("Model loaded successfully")
else:
    print("Model not found")
    sys.exit(1)
    
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
###################################################################################
#Predict using smooth blending

# Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
image = np.array(Image.open(input_image))  #N-34-66-C-c-4-3.tif, N-34-97-D-c-2-4.tif
image = preprocessing_fn(image)

input_image = image

def predict_mask(img_batch_subdiv):
    img_batch_subdiv = torch.from_numpy(img_batch_subdiv).to(DEVICE).float()
    img_batch_subdiv = img_batch_subdiv.permute(0,3,1,2)
    img_batch_subdiv = model(img_batch_subdiv)
    img_batch_subdiv = img_batch_subdiv.detach().squeeze().cpu().numpy()
    # print(img_batch_subdiv.shape) #(55, 7, 256, 256)
    # Convert pred_mask from `CHW` format to `HWC` format
    img_batch_subdiv = np.transpose(img_batch_subdiv,(0,2,3,1))
    
    return  img_batch_subdiv

predictions_smooth = predict_img_with_smooth_windowing(
    input_image,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=n_classes,
    pred_func=(
        lambda img_batch_subdiv: predict_mask((img_batch_subdiv))
    )
)

final_prediction = np.argmax(predictions_smooth, axis=2)


# make a color map of fixed colors
cmap = colors.ListedColormap(['cyan', 'yellow', 'magenta', 'green', 'blue', 'white', 'black'])
bounds=[0,1,2,3,4,5,6,7]
norm = colors.BoundaryNorm(bounds, cmap.N)



#Save prediction and original mask for comparison
# plt.imsave('pred_segmented_deepglobe_v4.png', final_prediction)
# plt.imsave('test_deepglobe/N-34-66-C-c-4-3.tif_mask.jpg', original_mask)
final_prediction = colour_code_segmentation(final_prediction, select_class_rgb_values)

# ValueError: Image RGB array must be uint8 or floating point; found int32
final_prediction = final_prediction.astype(np.uint8)
###################
image = Image.open(args.input_image)
plt.imsave(output_image_name, final_prediction)
# Plot the images
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(image)
plt.subplot(222)
# plt.title('Testing Label')
# plt.imshow(original_mask)
# plt.subplot(223)
plt.title('Prediction with smooth blending')
plt.imshow(final_prediction, cmap=cmap, norm=norm)
plt.show()