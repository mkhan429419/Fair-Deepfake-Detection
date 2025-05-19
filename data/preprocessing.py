import cv2
import numpy as np
from facenet_pytorch import MTCNN

class FacePreprocessor:
    def __init__(self):
        self.face_detector = MTCNN(keep_all=True)
        # Add Haar Cascade as fallback
        self.haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def extract_face_mask(self, image):
        """Generate binary mask for face region with fallback detection"""
        boxes, _ = self.face_detector.detect(image)
        
        # Fallback to Haar cascade if MTCNN fails
        if boxes is None or len(boxes) == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            haar_faces = self.haar_detector.detectMultiScale(gray, 1.3, 5)
            if len(haar_faces) > 0:
                boxes = []
                for (x, y, w, h) in haar_faces:
                    boxes.append([x, y, x+w, y+h])
                boxes = np.array(boxes)
            else:
                return np.zeros(image.shape[:2], dtype=np.float32)
            
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 1.0
            
        return mask
        
    def preprocess_image(self, image_path):
        """Preprocess image and generate face mask"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate face mask
        face_mask = self.extract_face_mask(image)
        
        return image, face_mask