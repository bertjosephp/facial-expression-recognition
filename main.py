from FaceDetector import FaceDetector


if __name__ == "__main__":
    face_detector = FaceDetector(filepath='emotion_v2.keras')
    face_detector.run()