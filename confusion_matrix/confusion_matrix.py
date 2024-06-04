import os
import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Net import Net

def load_test_data(test_dir):
    test_images = []
    test_labels = []
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    label_map = {label: idx for idx, label in enumerate(emotion_labels)}

    for label in emotion_labels:
        img_dir = os.path.join(test_dir, label)
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                test_images.append(img)
                test_labels.append(label_map[label])
            else:
                print(f"Failed to load image: {img_path}")

    return test_images, test_labels

def evaluate_model(model, test_images, test_labels):
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    actual = []
    predicted = []

    for i, (img, label) in enumerate(zip(test_images, test_labels)):
        if img is None:
            print(f"Skipping image {i} because it is None.")
            continue

        # Preprocess the image
        processed_image = transform(img)

        with torch.no_grad():
            processed_image = processed_image.unsqueeze(0)
            outputs = model(processed_image)
            _, predicted_emotion = torch.max(outputs, 1)
            predicted_emotion = predicted_emotion.item()

        actual.append(label)
        predicted.append(predicted_emotion)

    # Generate confusion matrix
    cm = confusion_matrix(actual, predicted, labels=list(range(len(emotion_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png')
    plt.show()

# Load the model
filepath = 'emotion_v3.pth'  # Path to the model file
model = Net()
model.load_state_dict(torch.load(filepath))
model.eval()

# Load test data
test_dir = 'fer2013/test' 
test_images, test_labels = load_test_data(test_dir)

# Evaluate the model and generate confusion matrix
evaluate_model(model, test_images, test_labels)

