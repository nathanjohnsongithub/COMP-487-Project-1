## Constants throughout both of the early tier models
Data Preprocessing:
Images resized to 224 × 224
Converted to tensors
Normalized using ImageNet mean and standard deviation
random horizontal flip and random rotation to avoid overfitting.

# The dataset contains:
100,000 training images
20,000 test images
Two classes: fake and real
This dataset was chosen because it directly matches the project goal of detecting whether an image is real or AI-generated.

Both models used a 70-30 split on the data. 
70,000 training images
30,000 validation images
20,000 test images

## Model 1: For the baseline approach, I implemented a ResNet-18 model on the CIFake dataset

I chose the resNet-18 as the baseline because it is good for image classification.
I had to modify the last fully connected layer to have only 2 classes instead of the original 1000 classes. 

Training setup:
Loss function: CrossEntropyLoss
Optimizer: Adam
Learning rate: 1e-4
Scheduler: StepLR(step_size=3, gamma=0.1)
Batch size: 64
Epochs: 10

## Model 2: EfficientNet-B0
EfficientNet-B0 was chosen because it is designed to scale network depth, width, and resolution more efficiently than older CNN architectures. It is often more parameter-efficient and can perform very well on image classification tasks. I froze the extractor layers because the runtime was too much.
Evaluated the model using accuracy and loss

Training setup:
Loss function: CrossEntropyLoss
Optimizer: Adam
Learning rate: 1e-4
Scheduler: StepLR(step_size=3, gamma=0.1)
Batch size: 64
Epochs: 10


## EVALUATION
ResNet18 Model results
Best validation accuracy: 0.9830
Final validation accuracy stayed around 0.9829
Test performance
Test accuracy: 0.9818

Precision, Recall, and f1-score 
FAKE class
precision: 0.98
recall: 0.98
f1-score: 0.98
REAL class
precision: 0.98
recall: 0.98
f1-score: 0.98

This means ResNet18 performed very well on both classes and did not show much class bias.

EfficientNet-B0 Results

Validation accuracy improved gradually during training:
Epoch 1: 0.8519
Epoch 10: 0.8708
Best validation accuracy: about 0.8725
Test performance
Test accuracy: 0.8579

EfficientNet-B0 learned the task, but it performed much worse than ResNet18.

## Analysis of the results
Overall comparison
ResNet18
Test accuracy: 0.9818
Best validation accuracy: 0.9830

EfficientNet-B0
Test accuracy: 0.8579
Best validation accuracy: about 0.8725

ResNet18 performed much better than EfficientNet-B0 on this project.
In this project, I compared ResNet18 and EfficientNet-B0 for classifying images as real or AI-generated. Both models used the same CIFAKE dataset, the same image size, and similar preprocessing steps. ResNet18 achieved much better results, with a test accuracy of 98.18%, while EfficientNet-B0 achieved 85.79%. One major reason is that ResNet18 was fully trainable, while EfficientNet-B0 had its feature extractor frozen, so only the final classifier was updated. In addition, the ResNet notebook used a cleaner validation setup by removing augmentation from the validation data. These differences made ResNet18 more effective for this task.
