import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import segmentation_models_pytorch as smp

from helpers import colour_code_segmentation
from smooth_tiled_predictions import predict_img_with_smooth_windowing

app = Flask(__name__)

# Configurações
MODEL_PATH = 'best_model_v4.pth'
DATA_DIR = 'data'
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
PATCH_SIZE = 256
N_DIVISIONS = 2
N_CLASSES = 7
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar modelo
if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    print("Modelo carregado com sucesso")
else:
    raise FileNotFoundError("Modelo não encontrado")

class_dict = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

select_classes = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def predict_mask(img_batch_subdiv):
    img_batch_subdiv = torch.from_numpy(img_batch_subdiv).to(DEVICE).float()
    img_batch_subdiv = img_batch_subdiv.permute(0, 3, 1, 2)
    img_batch_subdiv = model(img_batch_subdiv)
    img_batch_subdiv = img_batch_subdiv.detach().squeeze().cpu().numpy()
    img_batch_subdiv = np.transpose(img_batch_subdiv, (0, 2, 3, 1))
    return img_batch_subdiv

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    input_image = Image.open(file.stream)
    input_image = np.array(input_image)
    input_image = preprocessing_fn(input_image)

    predictions_smooth = predict_img_with_smooth_windowing(
        input_image,
        window_size=PATCH_SIZE,
        subdivisions=N_DIVISIONS,
        nb_classes=N_CLASSES,
        pred_func=lambda img_batch_subdiv: predict_mask(img_batch_subdiv)
    )

    final_prediction = np.argmax(predictions_smooth, axis=2)
    final_prediction = colour_code_segmentation(final_prediction, select_class_rgb_values)
    final_prediction = final_prediction.astype(np.uint8)

    output_image = Image.fromarray(final_prediction)
    buf = BytesIO()
    output_image.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png', as_attachment=True, download_name='prediction.png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
