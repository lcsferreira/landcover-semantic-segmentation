from matplotlib import colors
from matplotlib.colors import ListedColormap
from collections import namedtuple
import numpy as np
import os
import imageio
from PIL import Image

def RGBtoOneHot(rgb, colorDict):
  arr = np.zeros(rgb.shape[:2]) ## rgb shape: (h,w,3); arr shape: (h,w)
  #imprime a matriz rgb
  for label, color in enumerate(colorDict.keys()):
    color = np.array(color)
    if label < len(colorDict.keys()):
      #o print mostra o array de booleanos, onde a posicao que for igual a cor, vira true
      arr[np.all(rgb == color, axis=-1)] = label
  
  return arr


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

#se não existir a pasta "data/masks_encoded", cria a pasta
if not os.path.exists('data/masks_encoded'):
    os.makedirs('data/masks_encoded')
    
def encode_masks(folder):
    for file in os.listdir(folder):
        if file.endswith('.png'):
            mask = np.array(Image.open(os.path.join(folder, file)))
            one_hot_mask = RGBtoOneHot(mask, color2label)
            imageio.imwrite(os.path.join('data/masks_encoded', file), one_hot_mask.astype(np.uint8))
    
encode_masks('data/masks')