# Middle-Tier Notebook

The middle-tier implementations are in:

- `middle_tier_model_resnet.ipynb` (baseline notebook)
- `middle_tier_model_efficientnet.ipynb` (advanced architecture variant)

It keeps the dataset's natural split:

- `data/Dataset/Train`
- `data/Dataset/Validation`
- `data/Dataset/Test`


Implementation details:

- The baseline notebook uses a simple TensorFlow/Keras CNN model.
- The EfficientNet notebook uses an EfficientNetB0 backbone with a custom classification head.
- It loads images directly from the local folder-based dataset.
- It keeps per-class sample caps in place so the first version stays practical to run.