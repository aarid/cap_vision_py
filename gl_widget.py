from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPen, QSurfaceFormat
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
import mediapipe as mp
import trimesh

# gl_widget.py
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPen, QSurfaceFormat
from OpenGL.GL import *
import numpy as np
import cv2
import mediapipe as mp
import trimesh


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        # Set OpenGL format
        format = QSurfaceFormat()
        format.setDepthBufferSize(24)
        format.setStencilBufferSize(8)
        format.setVersion(3, 3)
        format.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(format)

        super().__init__(parent)
        print("GLWidget initialized")

        # Basic attributes
        self.image = None
        self.landmarks = None
        self.show_mesh = True
        self.show_landmarks = False

        # Model attributes
        self.model = None
        self.model_vertices = None
        self.model_faces = None
        self.model_scale = 1.0
        self.model_position = [0.0, 0.0, 0.0]
        self.model_rotation = [0.0, 0.0, 0.0]

        # OpenGL attributes
        self.background_texture = None
        self.background_vao = None
        self.background_vbo = None

        # MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

    def initializeGL(self):
        # Basic OpenGL settings
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

        # Create and bind VAO for background quad
        self.background_vao = glGenVertexArrays(1)
        self.background_vbo = glGenBuffers(1)

        # Create texture for video background
        self.background_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.background_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.image is not None:
            painter = QPainter(self)
            painter.beginNativePainting()

            # Convert image to Qt format
            h, w, ch = self.image.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Scale image to widget size while maintaining aspect ratio
            scaled_image = qt_image.scaled(self.size(), Qt.KeepAspectRatio)

            # Calculate position to center the image
            x_offset = (self.width() - scaled_image.width()) // 2
            y_offset = (self.height() - scaled_image.height()) // 2

            # Draw the image
            painter.drawImage(x_offset, y_offset, scaled_image)

            # Only draw mesh and landmarks if we have valid landmarks
            if self.landmarks is not None:
                # Calculate scale factors
                scale_x = scaled_image.width() / w
                scale_y = scaled_image.height() / h

                if self.show_mesh:
                    # Draw mesh connections
                    painter.setPen(QPen(Qt.blue, 1))
                    for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                        start = self.landmarks[connection[0]]
                        end = self.landmarks[connection[1]]

                        # Scale and offset coordinates
                        x1 = int(start.x * w * scale_x) + x_offset
                        y1 = int(start.y * h * scale_y) + y_offset
                        x2 = int(end.x * w * scale_x) + x_offset
                        y2 = int(end.y * h * scale_y) + y_offset

                        painter.drawLine(x1, y1, x2, y2)

                if self.show_landmarks:
                    # Draw landmarks
                    painter.setPen(QPen(Qt.green, 3))
                    for landmark in self.landmarks:
                        # Scale and offset coordinates
                        x = int(landmark.x * w * scale_x) + x_offset
                        y = int(landmark.y * h * scale_y) + y_offset
                        painter.drawPoint(x, y)

            painter.endNativePainting()
            painter.end()

    def update_camera_image(self, image):
        self.image = image.copy()
        self.update()

    def update_model_position(self, landmarks):
        """Update face landmarks and repaint. landmarks can be None if no face is detected."""
        self.landmarks = landmarks  # Will be None if no face is detected
        self.update()

    def load_model(self, model_path):
        try:
            self.model = trimesh.load(model_path)
            self.model_vertices = self.model.vertices
            self.model_faces = self.model.faces
            print(f"Model loaded: {len(self.model_vertices)} vertices, {len(self.model_faces)} faces")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # Existing setter methods
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

