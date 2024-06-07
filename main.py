from FaceDetector import FaceDetector
from Net import Net

if __name__ == "__main__":
    face_detector = FaceDetector(filepath='model_checkpoint.pth')
    face_detector.run()