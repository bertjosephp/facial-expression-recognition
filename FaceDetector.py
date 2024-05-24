import cv2


class FaceDetector:

    def __init__(self, source=0):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.video_capture = cv2.VideoCapture(source)

    def detect_faces(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=4, minSize=(30, 30))
        return faces
    
    def draw_bounding_boxes(self, frame, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=4)
    
    def run(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            self.draw_bounding_boxes(frame, faces)

            cv2.imshow("Live Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()