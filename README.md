# facial-expression-recognition

## Overview

This application uses a facial expression recognition model to detect and classify facial expressions in real-time using a webcam. The core of the application is a Convolutional Neural Network (CNN) model, trained on the FER-2013 dataset, which includes 28,709 training images and 3,589 test images categorized into seven different expressions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. For more detailed information about the dataset, visit [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).

## Installation

### Prerequisites

Ensure you have Python 3.9+ installed on your system, along with a working webcam.

### Steps

1. Clone this repository:
```bash
git clone git@github.com:bertjosephp/facial-expression-recognition.git
cd facial-expression-recognition
```

2. Install the necessary Python packages:
```bash
pip install -r requirements.txt
```

## Model Training

The model has already been trained, and the weights are stored in the `model_checkpoint.pth` file. If you want to use the pre-trained model, you can skip the training steps and proceed directly to running the application.

However, if you prefer to train the model yourself, here are the steps to do so:

1. Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
2. Rename the downloaded dataset to `fer2013` and place it in the root directory of the cloned repository. Ensure the dataset structure in the `fer2013` folder is divided into `train` and `test` subfolders, each containing class-labeled subfolders of images.
3. Run the training script by executing:
```bash
python3 train_model.py
```
This script will utilize the dataset, perform training using the CNN model defined in `Net.py`, and save the best model as `model_checkpoint.pth` based on test accuracy.

### Notes on Training

- The `train_model.py` script initializes model training and testing, setting the device to CUDA if available, or Apple's Metal if running on MacOS with M1/M2 chips, otherwise it defaults to CPU.
- Model checkpoints are saved when there is an improvement in test accuracy.
- Training results (loss and accuracy graphs) are saved in the `./training_outputs` directory for review.

## Running the application

To run the application, execute the following command in the terminal:
```bash
python3 main.py
```

This will activate the webcam, and you should see a window displaying the live video feed. Faces detected in the video feed will be highlighted with green rectangles.

## Exiting the application

To exit the application, simply press the 'q' key while the video window is active.
