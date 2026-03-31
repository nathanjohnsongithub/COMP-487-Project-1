# Modern-Tier Models

This folder contains the modern-tier image classification work for the **Fake vs Real** task.

## Models
- EfficientNetB0
- ResNet101
- ResNet50

## Dataset
This work uses the project dataset stored in:

`data/Dataset/`

with the following structure:

- Train/Fake
- Train/Real
- Validation/Fake
- Validation/Real
- Test/Fake
- Test/Real

## Task
Binary image classification:
- **Fake** = AI-generated images
- **Real** = real images

## Notebooks
- `modern_tier_model_efficientnet.ipynb`
- `modern_tier_model_resnet.ipynb`
- `modern_tier_model_resnet50.ipynb`

## Workflow
Each notebook includes:
- imports
- dataset loading
- model creation
- training
- evaluation
- training/validation accuracy plot
- prediction examples on test images