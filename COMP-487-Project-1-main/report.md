# Project 1 Simple Written Report

For our project, we want to test how a model detects real vs AI-generated images of landscapes from 3 different datasets. Specifically, a dataset from 10 years ago, a dataset from 2-3 years ago, and a dataset that was created within the last year. 

## Mid-Tier models

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
