# Airbus Ship Detection

## Model Architecture
The solution involves assembling models, including using ResNet50 to classify images for ship presence. For segmentation purposes, the `segmentation_models` library was utilized, selecting U-Net with a ResNet50 pretrained backbone.

## Metrics and Loss Function
For this project, I used the Dice Coefficient and Intersection over Union (IoU) as metrics, and Dice Loss as the loss function. The Dice Coefficient and IoU are particularly effective for evaluating segmentation models as they consider the overlap between the predicted and ground truth masks, providing a more accurate measure of the model's performance in segmentation tasks. Dice Loss was chosen as it is directly related to the Dice Coefficient, helping to optimize the model for better overlap between predicted and actual segmentation masks.

## Hyperparameter Tuning for ResNet50
- **Learning Rate (LR):** 6e-5
- **Batch Size:** 38
- **Epochs:** 1

## Hyperparameter Tuning for Unet
- **Learning Rate (LR):** 1e-4
- **Batch Size:** 22
- **Epochs:** 22

## Current Results
- **Public score:** 0.51864
- **Private score:** 0.75956
- **Best Validation Result:** F1-score: 0.7848