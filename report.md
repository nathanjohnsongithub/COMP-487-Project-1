# Project 1 Simple Written Report

For our project, we want to test how a model detects real versus AI-generated images across 3 different datasets. Specifically, we compare a dataset from about 10 years ago, a dataset from about 2-3 years ago, and a dataset created within the last year.

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

- Train: 6,000 per class, 12,000 total
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

For the **Simple CNN** and **EfficientNetB0** models, the learning rate was `0.001` and training ran for `10` epochs.

For **ResNet50**, training was split into two stages:

- Stage 1: `5` epochs with the backbone frozen and learning rate `0.001`
- Stage 2: `5` additional epochs with the last 30 ResNet50 layers unfrozen and learning rate `0.00001`

Final evaluation results were:

| Model | Training Accuracy | Validation Accuracy | Testing Accuracy |
| --- | ---: | ---: | ---: |
| Simple CNN | 0.6791 | 0.6503 | 0.6463 |
| EfficientNetB0 | 0.7968 | 0.7813 | 0.6887 |
| ResNet50 | 0.7882 | 0.7190 | 0.6673 |

### 6. Analysis of Results

The transfer learning models performed better than the simple CNN on all reported metrics. This shows that pretrained ImageNet features are useful for the real-versus-fake image detection task.

The simple CNN reached around 64.6% test accuracy, which suggests that the task is difficult and that a small custom CNN is not strong enough to capture all of the visual patterns that distinguish real images from generated ones. Something to keep in mind is that the base value is 50% because its binary classification, so 64.6% is only mildly better

The ResNet50 model improved over the simple CNN and reached about 66.7% test accuracy. Its validation accuracy was also stronger than the baseline. This suggests that a deeper pretrained backbone helps, but the gain was smaller than expected given the size of the model. Even with data augmentation and a short fine-tuning stage, ResNet50 did not outperform EfficientNetB0 on this dataset. This might be because of overall task difficulty.

EfficientNetB0 gave the best overall performance, reaching about 68.9% test accuracy. It also had the highest validation accuracy, which suggests that it provided the best balance between feature quality and generalization for this task. However, the test accuracy is still lower than the validation accuracy, which points to some remaining generalization issues.

For these models, the test accuracy is lower than the training accuracy, which suggests they perform better on the data they were trained on than on unseen images. This means the models learned useful patterns from the training set, but some of those patterns did not transfer as well to the test set. In general, this kind of drop from training to test accuracy points to limited generalization and suggests the models may be overfitting or that the test data is slightly different or more difficult than the training data.

### 7. Conclusion

For the mid-tier portion of the project, EfficientNetB0 was the strongest model overall, while ResNet50 ranked second and still outperformed the simple CNN baseline. The experiments demonstrate that model choice matters significantly for fake-versus-real image classification. 

## Modern-Tier Models

### 1. Problem Definition

For the modern-tier experiments, the goal was to test stronger transfer-learning models on the project `Fake` versus `Real` classification task and see how well newer architectures performed when trained on a capped version of the dataset.

### 2. Dataset Selection

The modern-tier notebooks used the project dataset stored in:

- `data/Dataset/Train`
- `data/Dataset/Validation`
- `data/Dataset/Test`

Each split contains two classes:

- `Fake`
- `Real`

Like the mid-tier notebooks, the modern-tier notebooks used capped samples per class to keep training practical:

- Train: 6,000 per class, 12,000 total
- Validation: 1,500 per class, 3,000 total
- Test: 1,500 per class, 3,000 total

### 3. Preprocessing

The modern-tier models followed the same overall split-based workflow:

- Load images from the existing folder-based `Train`, `Validation`, and `Test` splits
- Keep the binary `Fake` and `Real` labels
- Use a fixed random seed for reproducibility
- Train and evaluate with the same capped sample sizes across models

The ResNet-based modern models also used light data augmentation:

- Random horizontal flip
- Random rotation (`0.08`)
- Random zoom (`0.10`)

### 4. Model Architecture Design

Three modern-tier models were tested.

#### Model 1: EfficientNetB0

This model used a pretrained **EfficientNetB0** backbone with `include_top=False`, ImageNet weights, global average pooling, dropout, and a dense softmax output layer.

It had:

- **4,052,133 total parameters**
- **2,562 trainable parameters**

#### Model 2: ResNet101

This model used a pretrained **ResNet101** backbone with light augmentation, global average pooling, dropout, and a dense softmax output layer.

It had:

- **42,662,274 total parameters**
- **4,098 trainable parameters**

#### Model 3: ResNet50

This model used a pretrained **ResNet50** backbone with the same light augmentation pattern, global average pooling, dropout, and a dense softmax output layer.

It had:

- **23,591,810 total parameters**
- **4,098 trainable parameters**

### 5. Training and Evaluation

All three modern-tier models used the same general training setup:

- Optimizer: Adam
- Learning rate: `0.001`
- Loss function: sparse categorical crossentropy
- Batch size: `32`
- Epochs: `5`

Final evaluation results were:

| Model | Training Accuracy | Validation Accuracy | Testing Accuracy |
| --- | ---: | ---: | ---: |
| EfficientNetB0 | 1.0000 | 0.9994 | 0.9983 |
| ResNet101 | 0.9997 | 0.9994 | 0.9997 |
| ResNet50 | 0.9995 | 0.9994 | 0.9987 |

### 6. Analysis of Results

The modern-tier models performed extremely well. All three models achieved near-perfect validation and test accuracy, which means the binary classification problem in this tier was much easier for these transfer-learning models than the mid-tier task.

Among the three, **ResNet101** was the strongest overall model, with a test accuracy of **99.97%**. ResNet50 also performed extremely well at **99.87%**, while EfficientNetB0 reached **99.83%**.

The differences between the models were very small, but the deeper ResNet101 had the best final performance. This suggests that on this modern-tier dataset, all three pretrained backbones extracted highly effective visual features and generalization was not a major problem.

### 7. Conclusion

For the modern-tier portion of the project, **ResNet101** performed best overall, although all three models reached almost perfect results. The modern-tier task appears highly learnable with pretrained CNN backbones.

## Final Summary

Across the full project, the results show that performance depended heavily on the dataset tier and on how difficult the fake-versus-real patterns were in each dataset.

- In the **early tier**, ResNet18 was best with **98.18%** test accuracy, and the models performed very strongly overall.
- In the **mid tier**, the task was much harder. EfficientNetB0 was best, but it only reached **68.87%** test accuracy.
- In the **modern tier**, all three transfer-learning models were extremely strong, with ResNet101 finishing first at **99.97%** test accuracy.

One important limitation is that the three tiers were not perfectly identical in content or difficulty. The early-tier dataset, mid-tier dataset, and modern-tier dataset do not appear to be equally challenging, so the raw accuracies are not a perfectly apples-to-apples comparison. Even so, the experiments still show a useful trend: model performance changes a lot depending on both the architecture and the specific fake-image generation style represented in the dataset.

Overall, transfer learning was consistently helpful. The weakest results came from the small custom CNN in the mid tier, while pretrained ResNet and EfficientNet models usually performed much better. The project therefore suggests that pretrained visual backbones are the most reliable approach for fake-versus-real image classification, but dataset choice and task difficulty matter just as much as model choice.
