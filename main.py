from FaceDetector import FaceDetector
from Net import Net

if __name__ == "__main__":
    face_detector = FaceDetector(filepath='emotion_v3.pth')
    face_detector.run()