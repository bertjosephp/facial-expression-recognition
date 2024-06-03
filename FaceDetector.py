import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from Net import Net


class FaceDetector:

    def __init__(self, source=0, filepath='emotion_v3.pth'):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.video_capture = cv2.VideoCapture(source)
        self.model = Net()
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def detect_face_regions(self, frame):
        return self.face_classifier.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
    
    def draw_bounding_boxes(self, frame, faces, emotions):
        for (x, y, w, h), emotion in zip(faces, emotions):
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=4)
            cv2.putText(frame, text=emotion, org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 255), thickness=4)

    def process_face_region(self, face_region):
        return self.transform(face_region)

    def predict_emotion(self, processed_face_region):
        with torch.no_grad():
            processed_face_region = processed_face_region.unsqueeze(0)
            outputs = self.model(processed_face_region)
            _, predicted = torch.max(outputs, 1)
            emotion_index = predicted.item()
            return self.emotion_labels[emotion_index]

    def run(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            predicted_emotions = []
            face_regions = self.detect_face_regions(frame)
            for (x, y, w, h) in face_regions:
                face_region = frame[y:y+h, x:x+w]
                processed_face_image = self.process_face_region(face_region)
                predicted_emotion = self.predict_emotion(processed_face_image)
                predicted_emotions.append(predicted_emotion)
            
            self.draw_bounding_boxes(frame, face_regions, predicted_emotions)

            cv2.imshow("Live Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()