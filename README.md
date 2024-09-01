

# Brain Tumor Classification Using Convolutional Neural Network (CNN)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project focuses on the classification of brain tumors from MRI images using a Convolutional Neural Network (CNN). The primary goal is to create a model that can accurately classify whether an MRI image shows a brain tumor or not.

## Dataset

The dataset used in this project consists of MRI images categorized into two classes:
1. Tumor
2. Non-Tumor

You can download the dataset from [Kaggle](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri).

### Dataset Structure

- `Training Data`: Contains MRI images used to train the model.
- `Testing Data`: Contains MRI images used to evaluate the model's performance.

## Model Architecture

The model used for this project is a Convolutional Neural Network (CNN) based on the DenseNet201 architecture. The model includes the following layers:

- **Convolutional Layers**: Extract features from the input images.
- **Pooling Layers**: Reduce the spatial dimensions of the features.
- **Dense Layers**: Classify the extracted features into the respective classes.

### Key Components

- **DenseNet201**: A pre-trained CNN model used as the base model.
- **Data Augmentation**: Techniques applied to prevent overfitting.
- **Softmax Activation**: Used in the final layer for multi-class classification.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/brain-tumor-classification-cnn.git
    ```
2. Navigate to the project directory:
    ```bash
    cd brain-tumor-classification-cnn
    ```
3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Train the Model**:
    ```bash
    python train.py
    ```
2. **Evaluate the Model**:
    ```bash
    python evaluate.py
    ```
3. **Predict on New Data**:
    ```bash
    python predict.py --image path/to/image.jpg
    ```

## Results

The model achieved the following performance metrics on the test dataset:

- **Accuracy**: 95%
- **Precision**: 94%
- **Recall**: 93%
- **F1-Score**: 93%

### Sample Predictions

- Image 1: Tumor (Prediction: Tumor, Confidence: 98%)
- Image 2: Non-Tumor (Prediction: Non-Tumor, Confidence: 96%)

## Future Work

- **Hyperparameter Tuning**: Explore different architectures and parameters to improve performance.
- **Deployment**: Deploy the model as a web service for real-time tumor detection.
- **Multi-Class Classification**: Extend the model to classify different types of brain tumors.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


