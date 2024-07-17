## Semantic Segmentation Landcover - DeepGlobe

This is a repository containing the code for my final project. In it you can train a convolutional neural network for semantic segmentation. The aim of segmentation is to be able to use masks to analyze LoRA radio signals

## Train Model

You can train a DeepLabV3+ model with the command below

```bash
python app.py data_dir="data" encoder="resnet101" encoder_weights="imagenet" activation="softmax2d" train=True train_epochs=5 model_name="deepGlobe_resnet101.pth"
```

## Predict Image

You can generate a mask for a satellite image of your choice with the command below

```bash
python predict_image.py --data-dir data --input-image tcc_image.jfif --output-image-name pred_resnet101.png --encoder resnet101 --weights imagenet --model-path deepGlobeResnet101.pth --patch-size 256 --n-divisions 2 --n-classes 7
```

## Documentation for each prompt example

#### Train Model

| Parameter         | Descrição                                                           |
| :---------------- | :------------------------------------------------------------------ |
| `data_dir`        | **Required:** MUST provide the images and masks folder              |
| `encoder`         | **Default:** resnet101; **Options:** resnet34, resnet50, resnet101; |
| `encoder_weights` | **Default:** imagenet                                               |
| `activation`      | **Default:** softmax2d; **Options:** sigmoid, none                  |
| `train`           | **Default:** True (boolean to train the model or not)               |
| `train_epochs`    | **Default:** 30                                                     |
| `model_name`      | **Required:** Must provide the model name to save or to load        |

#### Predict Image

| Parameter           | Descrição                                                                          |
| :------------------ | :--------------------------------------------------------------------------------- |
| `data-dir`          | **Required:** MUST provide the images and masks folder                             |
| `input-image`       | **Required:** MUST provide the image path to predict the mask                      |
| `output-image-name` | **Required:** MUST provide the image path to save (including image extension)      |
| `encoder`           | **Default:** resnet101; **Options:** resnet34, resnet50, resnet101;                |
| `weights `          | **Default:** imagenet                                                              |
| `model-path`        | **Required:** Must provide the model name to save or to load                       |
| `patch-size`        | **Default:** 256; (size of the patches to predict and blend smoothly)              |
| `n-divisions`       | **Default:** 2; (Minimal amount of overlap for windowing. Must be an even number.) |
| `n-classes`         | **Required:** MUST provide the number of classes the model can predict             |

## Authors

- [@lcsferreira](https://github.com/lcsferreira)
