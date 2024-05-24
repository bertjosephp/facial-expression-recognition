import cv2
import tensorflow as tf


class FaceDetector:

    def __init__(self, source=0, filepath='emotion_v2.keras'):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.video_capture = cv2.VideoCapture(source)
        self.model = tf.keras.models.load_model(filepath)

    def detect_faces(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
        return faces
    
    def draw_bounding_boxes(self, frame, faces, emotions):
        for (x, y, w, h), emotion in zip(faces, emotions):
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=4)
            cv2.putText(frame, text=emotion, org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 255), thickness=4)

    def predict_emotion(self, face_image):
        resized_face_image = cv2.resize(face_image, (48, 48))
        if resized_face_image.shape[2] == 3:
            resized_face_image = cv2.cvtColor(resized_face_image, cv2.COLOR_BGR2GRAY)
        pass
    
    def run(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            emotions = [self.predict_emotion(frame[y:y+h, x:x+w]) for (x, y, w, h) in faces]
            self.draw_bounding_boxes(frame, faces, emotions)

            cv2.imshow("Live Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()