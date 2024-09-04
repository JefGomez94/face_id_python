import sys
import cv2
import numpy as np
import dlib
from PIL import Image
import face_recognition_models
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class ReconocimientoFacialApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Cargar la imagen de referencia y codificar el rostro
        self.known_image = self.load_image_file("carlos.jpeg")   ###ingresa la imagen de referencia
        self.known_encoding = self.face_encodings(self.known_image)[0]

    def initUI(self):
        self.setWindowTitle('Reconocimiento Facial')

        # Crear el layout de la ventana
        layout = QVBoxLayout()

        # Crear una etiqueta para mostrar el video de la cámara
        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        # Botón para iniciar la cámara
        self.start_button = QPushButton('Iniciar Cámara')
        self.start_button.clicked.connect(self.start_camera)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

        # Crear un temporizador para actualizar el video
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Iniciar captura de video
        self.cap = cv2.VideoCapture(0)

    def load_image_file(self, file):
        return np.array(Image.open(file))

    def face_locations(self, img):
        detector = dlib.get_frontal_face_detector()
        return [self._rect_to_css(face) for face in detector(img, 1)]

    def _rect_to_css(self, rect):
        return rect.top(), rect.right(), rect.bottom(), rect.left()

    def face_encodings(self, face_image, known_face_locations=None, num_jitters=1):
        face_locations = known_face_locations or self.face_locations(face_image)
        pose_predictor = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())
        face_encoder = dlib.face_recognition_model_v1(face_recognition_models.face_recognition_model_location())
            
        return [np.array(face_encoder.compute_face_descriptor(face_image, pose_predictor(face_image, dlib.rectangle(left, top, right, bottom)), num_jitters)) 
                for top, right, bottom, left in face_locations]
    
    def start_camera(self):
        self.timer.start(10)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convertir el frame al formato RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            # Encontrar las ubicaciones de los rostros en el frame
            face_locs = self.face_locations(rgb_frame)
            # Codificar los rostros encontrados en el frame
            face_encodings_found = self.face_encodings(rgb_frame, face_locs)
                
            for (top, right, bottom, left), self.face_encoding in zip(face_locs, face_encodings_found):
                # Comparar los rostros detectados con el rostro de referencia
                matches = np.linalg.norm(np.array(self.known_encoding) - np.array(self.face_encoding)) <= 0.6
                    
                # Dibujar un rectángulo alrededor de los rostros detectados
                if matches:
                    color = (0, 255, 0)  # Verde si es un rostro coincidente
                    label = "Rostro Coincide"
                else:
                    color = (0, 0, 255)  # Rojo si no coincide
                    label = "Rostro Desconocido"
                    
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            # Convertir el frame de OpenCV al formato de imagen de PyQt
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
                
            # Mostrar la imagen en la etiqueta
            self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ReconocimientoFacialApp()
    window.show()
    sys.exit(app.exec_())