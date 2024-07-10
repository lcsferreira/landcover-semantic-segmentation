#imports
# %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
from PIL import Image
# %pip install segmentation-models-pytorch==0.2.0
# !pip install -q -U segmentation-models-pytorch albumentations > /dev/null
# !pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp

import sys

from helpers import get_preprocessing, get_training_augmentation, get_validation_augmentation, visualize, one_hot_encode, reverse_one_hot, colour_code_segmentation
from LandoverDataset import LandCoverDataset

#use cuda or cpu (uncomment to check if cuda is available)
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
 
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"ID of current CUDA device: {torch.cuda.current_device()}")
       
# print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

#get the data_dir from the params in command line
# Function to parse arguments
def parse_arguments(args):
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    return params

# Get the arguments
arguments = sys.argv[1:]

# Parse the arguments
params = parse_arguments(arguments)

# Check if 'data_dir' is provided
if 'data_dir' in params:
    DATA_DIR = params['data_dir']
    print(f"Data directory: {DATA_DIR}")
else:
    print("Data directory not provided. Exiting...")
    # stop the script
    sys.exit()
# DATA_DIR = 'data'

metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata_patches.csv'))
metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]

# Shuffle DataFrame
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

# Perform train-validation split
valid_df = metadata_df.sample(frac=0.25, random_state=42)
train_df = metadata_df.drop(valid_df.index)
len(train_df), len(valid_df)
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(valid_df)}")

print("-"*50)

class_dict = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)

# Useful to shortlist specific classes in datasets with large number of classes
select_classes = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

print('Selected classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)

dataset = LandCoverDataset(train_df, class_rgb_values=select_class_rgb_values)
random_idx = random.randint(0, len(dataset)-1)
image, mask = dataset[2]

#   check the image and mask
# visualize(
#     original_image = image,
#     ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
#     one_hot_encoded_mask = reverse_one_hot(mask)
# )

augmented_dataset = LandCoverDataset(
    train_df, 
    augmentation=get_training_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

random_idx = random.randint(0, len(augmented_dataset)-1)

# Different augmentations on image/mask pairs
# for idx in range(3):
#     image, mask = augmented_dataset[idx]
#     visualize(
#         original_image = image,
#         ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
#         one_hot_encoded_mask = reverse_one_hot(mask)
#     )
    
CLASSES = select_classes

# get hyperparameters from the command line
# Get the arguments
arguments = sys.argv[1:]

# Parse the arguments
params = parse_arguments(arguments)
print("-"*50)
print("Hyperparameters:")
# Check if 'encoder' is provided
if 'encoder' in params:
    ENCODER = params['encoder']
    print(f"Encoder: {ENCODER}")
else:
    print("Encoder not provided. Defaulting to 'resnet101'...")
    ENCODER = 'resnet101'
    
# Check if 'encoder_weights' is provided
if 'encoder_weights' in params:
    ENCODER_WEIGHTS = params['encoder_weights']
    print(f"Encoder weights: {ENCODER_WEIGHTS}")
else:
    print("Encoder weights not provided. Defaulting to 'imagenet'...")
    ENCODER_WEIGHTS = 'imagenet'
    
# Check if 'activation' is provided
if 'activation' in params:
    ACTIVATION = params['activation']
    print(f"Activation: {ACTIVATION}")
else:
    print("Activation not provided. Defaulting to 'softmax2d'...")
    ACTIVATION = 'softmax2d' 

# ENCODER = 'resnet101'
# ENCODER_WEIGHTS = 'imagenet'
# ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation

# create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Get train and val dataset instances
train_dataset = LandCoverDataset(
    train_df, 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

valid_dataset = LandCoverDataset(
    valid_df, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

if "batch_size" in params:
    BATCH_SIZE = int(params['batch_size'])
    print(f"Batch size: {BATCH_SIZE}")
else:
    print("Batch size not provided. Defaulting to 16...")
    BATCH_SIZE = 16
    
# Get train and val data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

if("train" in params):
    TRAIN = params['train']
    print(f"IS TRAINING: {TRAIN}")
else:
    print("Training not provided. Defaulting to 'False'...")
    TRAIN = False

if "train_epochs" in params:
    EPOCHS = int(params['train_epochs'])
    print(f"Training epochs: {EPOCHS}")
else:
    print("Training epochs not provided. Defaulting to 20...")
    EPOCHS = 20
    
# define loss function
loss = smp.utils.losses.DiceLoss()

# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# define optimizer
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.00001),
])

# define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

# load best saved model checkpoint from previous commit (if present)
model_name = params['model_name']
if os.path.exists(model_name):
    model = torch.load(model_name, map_location=DEVICE)
    print('Loaded pre-trained DeepLabV3+ model!')
else:
    print('No pre-trained model found, training from scratch!')
    
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)   

if TRAIN:

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, model_name)
            print('Model saved!')
            
# create test dataloader to be used with DeepLabV3+ model (with preprocessing operation: to_tensor(...))
test_dataset = LandCoverDataset(
    valid_df, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

test_dataloader = DataLoader(test_dataset)

# test dataset for visualization (without preprocessing augmentations & transformations)
test_dataset_vis = LandCoverDataset(
    valid_df,
    augmentation=get_validation_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

test_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

valid_logs = test_epoch.run(test_dataloader)
print('\nEvaluation on Test Data: ')
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")
            
train_logs_df = pd.DataFrame(train_logs_list)
valid_logs_df = pd.DataFrame(valid_logs_list)
train_logs_df.T

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('IoU Score', fontsize=20)
plt.title('IoU Score Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('iou_score_plot.png')
plt.show()

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Dice Loss', fontsize=20)
plt.title('Dice Loss Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('dice_loss_plot.png')
plt.show()