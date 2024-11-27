Dental Radiography Classification

A deep learning-based approach to classify dental radiography images into four categories: Fillings, Implant, Impacted Tooth, and Cavity.

Table of Contents

1. #overview
2. #requirements
3. #dataset
4. #model-architecture
5. #training
6. #evaluation
7. #usage
8. #contributing
9. #license

Overview

This repository contains a deep learning-based approach to classify dental radiography images. The model uses a DenseNet201 architecture and is trained on a dataset of labeled dental radiography images.

Requirements

- Python 3.8+
- TensorFlow 2.4+
- Keras 2.4+
- NumPy 1.20+
- Pandas 1.3+
- Matplotlib 3.4+
- Scikit-learn 1.0+
Dataset

The dataset used for this project is a collection of labeled dental radiography images. The dataset is divided into four categories: Fillings, Implant, Impacted Tooth, and Cavity.

Model Architecture

The model uses a DenseNet201 architecture with a custom classification head. The model is trained using a sparse categorical cross-entropy loss function and the Adam optimizer.

Training

The model is trained on the training dataset for 100 epochs with a batch size of 64. The model is evaluated on the validation dataset after each epoch.

Evaluation

The model is evaluated on the test dataset using metrics such as accuracy, precision, recall, and F1-score.

Usage

To use this repository, follow these steps:

1. Clone the repository using git clone.
2. Install the required dependencies using pip install -r requirements.txt.
3. Download the dataset and preprocess it according to the instructions in the data directory.
4. Train the model using the (link unavailable) script.
5. Evaluate the model using the (link unavailable) script.

Author: vaishnav muriki
Email: vaishpaa@gmail.com
