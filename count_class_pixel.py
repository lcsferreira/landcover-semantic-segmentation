import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plot

mask_dir = "data/masks"

classes = {
  "urban_land": [0,255,255],
  "agriculture_land": [255,255,0],
  "rangeland": [255,0,255],
  "rangeland": [255,0,255],
  "forest_land": [0,255,0],
  "water": [0,0,255],
  "barren_land": [255,255,255],
  "unkown": [0,0,0]
}

pixel_count = {class_name: 0 for class_name in classes}

# for mask_file in os.listdir(mask_dir):
#   print('Processing', mask_file)
#   mask_path = os.path.join(mask_dir, mask_file)
#   mask = np.array(Image.open(mask_path))
  
#   for class_name, color in classes.items():
#     mask_class = cv2.inRange(mask, np.array(color), np.array(color))
    
#     pixel_count[class_name] += np.count_nonzero(mask_class)
    

# total_pixels = sum(pixel_count.values())

# percentages = {class_name: (count / total_pixels) * 100 for class_name, count in pixel_count.items()}

# for class_name, percentage in percentages.items():
#   print(f'{class_name}: {percentage:.2f}%')

# image1 = "previsoes_VALID/pred_108490_sat.jpg"
# image2 = "previsoes_VALID_nosmooth/pred_nosmooth_108490_sat.jpg"

mask_file = 'previsoes_VALID/pred_108490_sat.jpg'
mask = np.array(Image.open(mask_file))

# remove a quarta coluna
mask = mask[:, :, :3]

plot.imshow(mask)
plot.show()

# conta os pixels de cada classe no new_mask
pixel_count = {class_name: 0 for class_name in classes}
for class_name, color in classes.items():
  mask_class = cv2.inRange(mask, np.array(color), np.array(color))
  pixel_count[class_name] = np.count_nonzero(mask_class)
  
print(pixel_count)
total_pixels = sum(pixel_count.values())
percentages = {class_name: (count / total_pixels) * 100 for class_name, count
in pixel_count.items()}
for class_name, percentage in percentages.items():
  print(f'{class_name}: {percentage:.2f}%')