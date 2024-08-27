sample_preds_folder = 'data/sample_predictions/'
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)
    

def calcular_dissimilaridade(gt_mask, pred_mask):
    """
    Calcula a dissimilaridade normalizada entre a máscara ground truth (gt_mask) e a máscara predita (pred_mask)
    usando a distância euclidiana.

    Parâmetros:
    gt_mask (numpy array): A máscara ground truth no formato HWC.
    pred_mask (numpy array): A máscara predita no formato HWC.

    Retorna:
    dissimilarity_normalized (float): A dissimilaridade normalizada entre as duas máscaras.
    """
    # Garantir que as máscaras têm a mesma forma
    assert gt_mask.shape == pred_mask.shape, "As máscaras devem ter a mesma forma."

    # Calcular a diferença entre as máscaras
    difference = gt_mask.astype(float) - pred_mask.astype(float)
    
    # Calcular a distância euclideana
    dissimilarity = np.sqrt(np.sum(difference ** 2))
    
    # Normalizar pela quantidade de pixels
    dissimilarity_normalized = dissimilarity / gt_mask.size
    
    return dissimilarity_normalized


# create logfile to store the results of dissimilarity
logfile = open('dissimilarity_log.txt', 'w')
logfile.write('Dissimilarity results\n')

for idx in range(len(test_dataset)):

    image, gt_mask = test_dataset[idx]
    image_vis = test_dataset_vis[idx][0].astype('uint8')
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # Predict test image
    pred_mask = model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask,(1,2,0))
    # Get prediction channel corresponding to foreground
    pred_urban_land_heatmap = pred_mask[:,:,select_classes.index('urban_land')]
    pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
    # Convert gt_mask from `CHW` format to `HWC` format
    gt_mask = np.transpose(gt_mask,(1,2,0))
    gt_mask = colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values)
    # calculate dissimilarity
    dissimilarity = calcular_dissimilaridade(gt_mask, pred_mask)
    logfile.write(f"Sample {idx} dissimilarity: {dissimilarity}\n")
    # Save sample predictions
    cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])


logfile.close()

#calculate the average dissimilarity
dissimilarity_values = []
with open('dissimilarity_log.txt', 'r') as file:
    for line in file:
        if 'Sample' in line:
            dissimilarity_values.append(float(line.split(':')[-1].strip()))
            
average_dissimilarity = sum(dissimilarity_values) / len(dissimilarity_values)

print(f'Average dissimilarity: {average_dissimilarity}')