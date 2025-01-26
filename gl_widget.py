from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
import mediapipe as mp
import numpy as np


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.landmarks = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        # Visualization flags
        self.show_mesh = True
        self.show_landmarks = False

        # Model parameters
        self.model_scale = 1.0
        self.model_position = [0.0, 0.0, 0.0]

    def update_camera_image(self, image):
        """Update the camera image and repaint"""
        self.image = image.copy()
        self.update()

    def update_model_position(self, landmarks):
        """Update face landmarks and repaint"""
        self.landmarks = landmarks
        self.update()

    def paintEvent(self, event):
        if self.image is None:
            return

        # Convert image to Qt format
        h, w, ch = self.image.shape
        bytes_per_line = ch * w
        image = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Create painter
        painter = QPainter(self)

        # Draw the image scaled to widget size
        scaled_image = image.scaled(self.size(), Qt.KeepAspectRatio)
        x = (self.width() - scaled_image.width()) // 2
        y = (self.height() - scaled_image.height()) // 2
        painter.drawImage(x, y, scaled_image)

        if self.landmarks:
            # Scale factors for drawing
            scale_x = scaled_image.width() / w
            scale_y = scaled_image.height() / h

            if self.show_landmarks:
                # Draw landmarks
                painter.setPen(QPen(Qt.green, 2))
                for landmark in self.landmarks:
                    x_pos = int(x + landmark.x * scaled_image.width())
                    y_pos = int(y + landmark.y * scaled_image.height())
                    painter.drawPoint(x_pos, y_pos)

            if self.show_mesh:
                # Draw face mesh connections
                painter.setPen(QPen(Qt.blue, 1))
                for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                    start = self.landmarks[connection[0]]
                    end = self.landmarks[connection[1]]

                    x1 = int(x + start.x * scaled_image.width())
                    y1 = int(y + start.y * scaled_image.height())
                    x2 = int(x + end.x * scaled_image.width())
                    y2 = int(y + end.y * scaled_image.height())

                    painter.drawLine(x1, y1, x2, y2)

                # Draw contours
                painter.setPen(QPen(Qt.red, 2))
                for connection in mp.solutions.face_mesh.FACEMESH_CONTOURS:
                    start = self.landmarks[connection[0]]
                    end = self.landmarks[connection[1]]

                    x1 = int(x + start.x * scaled_image.width())
                    y1 = int(y + start.y * scaled_image.height())
                    x2 = int(x + end.x * scaled_image.width())
                    y2 = int(y + end.y * scaled_image.height())

                    painter.drawLine(x1, y1, x2, y2)

        painter.end()

    def set_show_mesh(self, show):
        self.show_mesh = show
        self.update()

    def set_show_landmarks(self, show):
        self.show_landmarks = show
        self.update()

    def set_scale(self, scale):
        self.model_scale = scale
        self.update()

    def set_position(self, axis, value):
        if axis == 'X':
            self.model_position[0] = value
        elif axis == 'Y':
            self.model_position[1] = value
        elif axis == 'Z':
            self.model_position[2] = value
        self.update()

    def set_cap_model(self, index):
        print(f"Switching to cap model {index}")
        self.update()

    def load_model(self, model_path):
        """Placeholder for 3D model loading"""
        print(f"Loading model from: {model_path}")