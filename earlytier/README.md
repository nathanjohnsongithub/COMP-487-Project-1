## Implementation 1: For the baseline approach, I implemented a ResNet-18 model on the CIFake dataset

I chose the resNet-18 as the baseline because it is good for image classification.
I had to modify the last fully connected layer to have only 2 classes instead of the original 1000 classes. 

Hyperparameters
Batch size: 64
Epochs: 10 (Took 50 minutes for 10 epochs)
Learning rate: 1e-4
Image size: 224 × 224 (standard input size for ResNet models
Validation split: 20%
Adam optimizer


Data Preprocessing:
Resize images to 224x224
normalization
random horizontal flip and random rotation to avoid overfitting. 


##Evaluation:
Evaluated the model using accuracy and loss

