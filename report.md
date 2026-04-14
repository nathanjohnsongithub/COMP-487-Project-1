# Project 1 Simple Written Report

For our project, we want to test how a model detects real versus AI-generated images across 3 different datasets. Specifically, we compare a dataset from about 10 years ago, a dataset from about 2-3 years ago, and a dataset created within the last year.

# Presentation link

https://docs.google.com/presentation/d/10-KqbGpKRMC_cuTnW5cPd6c-GulO3G73rOfHsknJU9s/edit?usp=sharing

## Early-Tier Models

### 1. Problem Definition

For the early-tier experiments, the goal was to test how well older-style fake-versus-real image classification could be learned from a large binary dataset. This tier was meant to represent earlier AI-generated image data and to compare a strong pretrained CNN baseline against a more modern efficient architecture.

### 2. Dataset Selection

The early-tier models used the **CIFAKE** dataset. The dataset contains:

- 100,000 training images
- 20,000 test images
- Two classes: `fake` and `real`

Both models then used a `70/30` split of the training portion for model development:

- 70,000 training images
- 30,000 validation images
- 20,000 test images

This dataset was chosen because it directly matched the project goal of identifying whether an image is authentic or AI-generated.

### 3. Preprocessing

The preprocessing pipeline was the same for both early-tier models:

- Resize all images to `224 x 224`
- Convert images to tensors
- Normalize using the ImageNet mean and standard deviation
- Apply random horizontal flips and random rotation to reduce overfitting

### 4. Model Architecture Design

Two early-tier models were tested.

#### Model 1: ResNet18

The baseline early-tier model used **ResNet18** pretrained on ImageNet. The final fully connected layer was replaced so the network predicted only two classes instead of the original 1000 ImageNet classes.

This model was chosen because ResNet18 is a standard image-classification architecture that is deep enough to learn useful visual structure while still being practical to train.

#### Model 2: EfficientNet-B0

The second model used **EfficientNet-B0**. It was selected because EfficientNet scales depth, width, and resolution more efficiently than many older CNN designs and is usually strong for image-classification tasks. In this notebook, the feature extractor layers were frozen to keep runtime manageable, and only the classifier head was trained.

### 5. Training and Evaluation

Both early-tier models used the same main training setup:

- Loss function: `CrossEntropyLoss`
- Optimizer: Adam
- Learning rate: `1e-4`
- Scheduler: `StepLR(step_size=3, gamma=0.1)`
- Batch size: `64`
- Epochs: `10`

Final evaluation results were:

| Model | Best Validation Accuracy | Final Validation Accuracy | Testing Accuracy |
| --- | ---: | ---: | ---: |
| ResNet18 | 0.9830 | 0.9829 | 0.9818 |
| EfficientNet-B0 | 0.8725 | 0.8708 | 0.8579 |

For the ResNet18 model, the classification report also showed very balanced class performance:

- Fake precision: `0.98`
- Fake recall: `0.98`
- Fake F1-score: `0.98`
- Real precision: `0.98`
- Real recall: `0.98`
- Real F1-score: `0.98`

### 6. Analysis of Results

The early-tier results were very strong overall. ResNet18 reached about **98.2%** test accuracy and clearly outperformed EfficientNet-B0. This suggests that the earlier dataset was easier for the model to separate into fake and real classes, or that the patterns distinguishing the classes were more visually consistent.

EfficientNet-B0 still learned the task well, but it performed much worse than ResNet18. A likely reason is that its feature extractor was frozen, so the model had much less ability to adapt to the dataset than the fully trainable ResNet18 model.

Another likely factor is the validation setup. The ResNet18 notebook used a cleaner validation pipeline, while the EfficientNet setup had fewer learnable components. Together, those choices made ResNet18 the stronger early-tier model.

### 7. Conclusion

For the early-tier portion of the project, **ResNet18** was the best model by a large margin. It achieved the highest validation and testing accuracy and showed balanced precision and recall across both classes.

## Mid-Tier Models

### 1. Problem Definition

We wanted to see how good CNN are ate recognizing fake vs real images. FOr the Mid-tier we're testing fake images generated from a couple years ago, to see if its easier for models to detect AI generated or fake images from 3 years ago compared to even older and the newest versions.

### 2. Dataset Selection

For the mid-tier experiments, we used the local project copy of the *deepfake and real images* dataset that I found on [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data). This dataset is a subset of a dataset from the paper [OpenForensics: Multi-Face Forgery Detection And Segmentation In-The-Wild Dataset](https://zenodo.org/records/5528418#.YpdlS2hBzDd). The dataset is all deepfake images of people with there faces swapped and then real images. It contains various backgrounds and multiple people of various ages, genders, poses, positions, and face occlusions.

The dataset here is using Generative Adversarial Network or GAN in the format of StyleGAN and/or ALAE. The faces are generated using  latent-vector manipulation then face-swapped into real images. This is what is known as a deepfake and it's very difficult to detect with the naked human eye. GANs are quite good at controlled face generation compared to diffusion models, which is why they we're used in this dataset.

The entire pipeline is something like this

- Generate identity using StyleGAN / ALAE
- Insert into real image via face swapping
- Then Blend with Poisson + landmarks

### 3. Preprocessing

The Kaggle version of the dataset came in the format below. 

- `data/Dataset/Train`
- `data/Dataset/Validation`
- `data/Dataset/Test`

Each split contains two classes: `Fake` and `Real`.

The full local dataset contains:

- Train: 140,002 images
- Validation: 39,428 images
- Test: 10,905 images

To keep training practical on my computer hardware, the notebooks cap the number of images used per class to

- Train: 10,000 per class, 20,000 total
- Validation: 1,500 per class, 3,000 total
- Test: 1,500 per class, 3,000 total

The preprocessing steps were:

- Load images directly from the folder-based dataset split
- Resize all images to `128 x 128`
- Convert images to RGB arrays
- Normalize pixel values to the range `[0, 1]`
- Encode the class labels as integers for `Fake` and `Real`

### 4. Model Architecture Design

Three mid-tier models were tested.

#### Model 1: Simple CNN

The first model is a straightforward convolutional neural network used as a baseline. Its structure is:

- `Conv2D(32)` + max pooling
- `Conv2D(64)` + max pooling
- `Conv2D(128)`
- Global average pooling
- Dropout (`0.3`)
- Dense softmax output layer

This model has **93,506 trainable parameters**. It is lightweight and easy to train, but it has limited representational power compared with transfer learning models.

#### Model 2: EfficientNetB0

The second model uses **EfficientNetB0** with pretrained ImageNet weights as the feature extractor. The architecture is:

- Input image
- Rescaling layer
- Frozen EfficientNetB0 backbone (`include_top=False`)
- Global average pooling
- Dropout (`0.3`)
- Dense softmax output layer

This model has **4,052,133 total parameters**, but only **2,562 trainable parameters** because the pretrained backbone was frozen during training. This approach uses transfer learning to provide stronger visual features when compared to the simple CNN.

#### Model 3: ResNet50

The third model uses **ResNet50** with pretrained ImageNet weights and a slightly larger classification head. It also adds light data augmentation and a short fine-tuning stage. The architecture is:

- Input image
- Data augmentation:
  - Random horizontal flip
  - Random rotation (`0.08`)
  - Random zoom (`0.10`)
- Rescaling layer
- ResNet50 preprocessing
- Frozen ResNet50 backbone (`include_top=False`)
- Global average pooling
- Batch normalization
- Dropout (`0.4`)
- Dense layer (`256`, ReLU)
- Dropout (`0.4`)
- Dense softmax output layer

This model has **24,120,962 total parameters**. During the first stage, only the classification head was trainable, for **529,154 trainable parameters**. After that, the last 30 layers of the ResNet50 backbone were unfrozen for a short fine-tuning stage.

### 5. Training and Evaluation

All three models were trained with the same general setup:

- Optimizer: Adam
- Loss function: sparse categorical crossentropy
- Batch size: `32`
- Metric: accuracy

For the **Simple CNN** and **EfficientNetB0** models, the learning rate was `0.001` and training ran for `15` epochs.

For **ResNet50**, training was split into two stages:

- Stage 1: `5` epochs with the backbone frozen and learning rate `0.001`
- Stage 2: `10` additional epochs with the last 30 ResNet50 layers unfrozen and learning rate `0.00001`

Final evaluation results were:

| Model | Training Accuracy | Validation Accuracy | Testing Accuracy |
| --- | ---: | ---: | ---: |
| Simple CNN | 0.8138 | 0.7470 | 0.7137 |
| EfficientNetB0 | 0.7893 | 0.7670 | 0.6933 |
| ResNet50 | 0.8543 | 0.7787 | 0.6803 |

### 6. Confusion Matrices and Additional Metrics

After running the three mid-tier notebooks, I also recorded the confusion-matrix results and additional binary-classification metrics. In these notebooks, `Real` was treated as the positive class.

| Model | Precision | Recall | Specificity | F1 Score | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple CNN | 0.6760 | 0.8207 | 0.6067 | 0.7413 | 0.7137 |
| EfficientNetB0 | 0.7680 | 0.5540 | 0.8327 | 0.6437 | 0.6933 |
| ResNet50 | 0.9068 | 0.4020 | 0.9587 | 0.5570 | 0.6803 |

The confusion matrices make the tradeoffs between the models much clearer than accuracy alone.

- The Simple CNN gave the best overall test accuracy at 71.37% and the best F1 score at 0.7413. Its confusion matrix was TP = 1231, TN = 910, FP = 590, and FN = 269. That means it found most of the real images well, but it also mislabeled a large number of fake images as real.
- EfficientNetB0 was more balanced on the fake-image side. Its confusion matrix was TP = 831, TN = 1249, FP = 251, and FN = 669. It had lower recall than the Simple CNN, but much better specificity, so it was more cautious about calling an image real.
- ResNet50 was the most conservative model. Its confusion matrix was TP = 603, TN = 1438, FP = 62, and FN = 897. This explains why it had extremely high precision (0.9068) and specificity (0.9587) but much lower recall (0.4020). In other words, when it predicted Real it was usually correct, but it missed many real images.

These metrics show that the choice of best model depends on what matters most. If the goal is strongest overall balance between precision and recall, the Simple CNN was actually the strongest mid-tier model after running the notebooks. If the goal is to avoid falsely labeling fake images as real, ResNet50 was strongest because it had the fewest false positives.

### 7. Analysis of Results

The simple CNN produced the highest test accuracy, while EfficientNetB0 placed second and ResNet50 placed third on the held-out test split.

The **Simple CNN** improved the most and reached **71.37%** test accuracy. Even though it is the smallest model, it generalized better than the two transfer-learning models. Its recall for the `Real` class was also the strongest, which helped push its F1 score above the others.

The **EfficientNetB0** model remained competitive and had the highest validation accuracy at **76.70%**, but its test accuracy dropped to **69.33%**. This suggests it learned useful features from the training and validation data, but those patterns did not transfer quite as well to the test set as the simple CNN.

The **ResNet50** model achieved **68.03%** test accuracy. Its metrics show a very different behavior from the other two models: it was highly precise when predicting `Real`, but it failed to recover many of the real images in the test set. This made it a stricter model, but not the best balanced one overall.

For all three models, training accuracy was still higher than test accuracy. That suggests the task remains difficult and that each model still struggles to generalize perfectly to unseen images. Even so, all three models stayed clearly above the 50% random baseline for a binary classification problem.

### 8. Conclusion

For the mid-tier portion of the project, the results show that the **Simple CNN** was the strongest overall model on the test set, followed by **EfficientNetB0**, then **ResNet50**. The confusion-matrix results also showed that these models were making different kinds of mistakes, so accuracy alone did not tell the full story. 

## Modern-Tier Models

### 1. Problem Definition

For the modern-tier experiments, the goal was to test how well newer transfer-learning image classifiers could distinguish real images from AI-generated images produced by recent diffusion-based systems. This tier represents the newest generation of fake images in the project, so it is intended to be the most modern and visually advanced setting. We wanted to compare three strong pretrained CNN backbones and see which one performed best on this modern fake-versus-real classification task.

### 2. Dataset Selection

For the modern-tier experiments, we used the Synthbuster dataset. This dataset contains AI-generated images from recent text-to-image systems including:
	•	DALL·E 2
	•	DALL·E 3
	•	Adobe Firefly
	•	Glide
	•	Midjourney v5
	•	Stable Diffusion 1.3
	•	Stable Diffusion 1.4
	•	Stable Diffusion 2
	•	Stable Diffusion XL

The original Synthbuster release mainly contains fake/generated images, so for the real class we paired it with a separate natural image collection. The real images were taken from a natural image dataset containing categories such as airplane, car, cat, dog, flower, fruit, motorbike, and person.

The final local modern-tier dataset used in our experiments contained:
	•	Fake images: 9,003
	•	Real images: 6,899

These were split into:
	•	Train: 10,923 images total
	•	Validation: 2,341 images total
	•	Test: 2,635 images total

This dataset was selected because it directly matches the project goal of testing detection performance on the newest AI-generated images.

### 3. Preprocessing

The preprocessing pipeline for the modern-tier models was:
	•	Load images directly from folder-based train, validation, and test directories
	•	Resize all images to 224 × 224
	•	Convert images to RGB tensors
	•	Use binary labels for Fake and Real
	•	Apply light data augmentation during training:
	  •	random horizontal flip
	  •	random rotation
	  •	random zoom

The dataset folders used the structure:
	•	data/modern_dataset/Train
	•	data/modern_dataset/Validation
	•	data/modern_dataset/Test

with two classes inside each split:
	•	Fake
	•	Real
  
### 4. Model Architecture Design

Three modern-tier models were tested.

#### Model 1: EfficientNetB0

The first modern-tier model used EfficientNetB0 with pretrained ImageNet weights as a frozen feature extractor. The architecture was:
	•	Input image
	•	Data augmentation layer
	•	EfficientNetB0 preprocessing
	•	Frozen EfficientNetB0 backbone (include_top=False)
	•	Global average pooling
	•	Dropout (0.3)
	•	Dense sigmoid output layer

This model was chosen because EfficientNetB0 is compact, efficient, and usually performs strongly on image classification tasks while keeping training practical.

#### Model 2: ResNet50

The second model used ResNet50 with pretrained ImageNet weights. The architecture was:
	•	Input image
	•	Data augmentation layer
	•	ResNet50 preprocessing
	•	Frozen ResNet50 backbone (include_top=False)
	•	Global average pooling
	•	Dropout (0.3)
	•	Dense sigmoid output layer

This model was chosen because ResNet50 is a deeper pretrained CNN that can learn stronger visual representations than smaller baselines.

#### Model 3: MobileNetV2

The third model used MobileNetV2 with pretrained ImageNet weights. The architecture was:
	•	Input image
	•	Data augmentation layer
	•	MobileNetV2 preprocessing
	•	Frozen MobileNetV2 backbone (include_top=False)
	•	Global average pooling
	•	Dropout (0.3)
	•	Dense sigmoid output layer

This model was selected because MobileNetV2 is lightweight and efficient while still providing strong transfer-learning performance.

### 5. Training and Evaluation

All three modern-tier models used the same general training setup:

- Optimizer: Adam
- Learning rate: `0.001`
- Loss function: sparse categorical crossentropy
- Batch size: `32`
- Epochs: `5`
- Metric: accuracy

Final evaluation results were:

| Model | Testing Accuracy |
| --- | ---: |
| EfficientNetB0 | 0.9806 |
| ResNet50 | 0.9856 |
| MobileNetV2 | 0.9784 |

### 6. Analysis of Results

The modern-tier models performed extremely strongly overall. All three pretrained CNN backbones achieved very high test accuracy, showing that transfer learning worked very well for this dataset.

Among the three models, **ResNet50** performed the best with about **98.56%** test accuracy. This suggests that the deeper residual architecture was able to capture the most useful visual patterns for separating real images from modern AI-generated images.

**EfficientNetB0** also performed very well, reaching about **98.06%** test accuracy. Its result was only slightly below ResNet50, which shows that it remained a very strong and efficient model for this task.

**MobileNetV2** had the lowest result of the three, but it still achieved about **97.84%** test accuracy, which is still excellent. This shows that even a lighter pretrained backbone can perform strongly on modern fake-versus-real image classification.

Overall, the modern-tier results were much stronger than the mid-tier results and were also highly competitive with the early-tier results. This suggests that, for this particular dataset split and preprocessing setup, the modern-tier task was highly learnable with pretrained CNN backbones.

### 7. Conclusion

For the modern-tier portion of the project, ResNet50 performed best overall, although all three models reached very strong results. The experiments show that pretrained CNN backbones are highly effective for detecting real versus modern AI-generated images on this dataset.

## Final Summary

Across the full project, the results showed that performance depended heavily on the dataset tier and on how difficult the fake-versus-real patterns were in each dataset.
	•	In the **early tier**, ResNet18 performed best with **98.18%** test accuracy, and the models performed very strongly overall.
	•	In the **mid tier**, the task was much harder. EfficientNetB0 performed best there, reaching **68.87%** test accuracy.
	•	In the modern tier, all three transfer-learning models performed extremely well, with **ResNet50** finishing first at **98.56%** test accuracy.

One important limitation is that the three tiers were not perfectly identical in content or difficulty. The early-tier, mid-tier, and modern-tier datasets do not appear to be equally challenging, so the raw accuracies are not a perfectly apples-to-apples comparison. Even so, the experiments still show a useful trend: model performance changes a lot depending on both the architecture and the type of fake-image generation represented in the dataset.

Overall, transfer learning was consistently helpful. The weakest results came from the small custom CNN in the mid tier, while pretrained ResNet, EfficientNet, and MobileNet models generally performed much better. The project therefore suggests that pretrained visual backbones are a reliable approach for fake-versus-real image classification, but dataset difficulty and generation style matter just as much as model choice.
