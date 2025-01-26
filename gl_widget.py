
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
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.shader_program = None

        # Shader source code
        self.vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
        }
        """

        self.fragment_shader_source = """
        #version 330 core
        out vec4 FragColor;

        void main() {
            FragColor = vec4(0.7, 0.7, 0.7, 1.0);  // Gris
        }
        """

        # Matrices for 3D rendering
        self.projection = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, -2],
            [0, 0, -1, 0]
        ], dtype=np.float32)

        self.view = np.eye(4, dtype=np.float32)
        self.view[2, 3] = -5  # Camera position

        # MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

    def initializeGL(self):
        print("initializeGL called")

        # Basic OpenGL settings
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

        try:
            # Create and compile vertex shader
            vertex_shader = glCreateShader(GL_VERTEX_SHADER)
            glShaderSource(vertex_shader, self.vertex_shader_source)
            glCompileShader(vertex_shader)
            if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
                print("Vertex shader compilation failed:", glGetShaderInfoLog(vertex_shader))
                return

            # Create and compile fragment shader
            fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
            glShaderSource(fragment_shader, self.fragment_shader_source)
            glCompileShader(fragment_shader)
            if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
                print("Fragment shader compilation failed:", glGetShaderInfoLog(fragment_shader))
                return

            # Create shader program
            self.shader_program = glCreateProgram()
            glAttachShader(self.shader_program, vertex_shader)
            glAttachShader(self.shader_program, fragment_shader)
            glLinkProgram(self.shader_program)

            # Check program linking
            if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
                print("Shader program linking failed:", glGetProgramInfoLog(self.shader_program))
                return

            # Clean up shaders
            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)

            print("Shaders compiled successfully")

        except Exception as e:
            print(f"Error in initializeGL: {e}")

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.image is not None:
            painter = QPainter(self)
            painter.beginNativePainting()

            # Draw video frame
            h, w, ch = self.image.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(self.size(), Qt.KeepAspectRatio)
            x_offset = (self.width() - scaled_image.width()) // 2
            y_offset = (self.height() - scaled_image.height()) // 2
            painter.drawImage(x_offset, y_offset, scaled_image)

            # Draw face mesh and landmarks
            if self.landmarks is not None:
                scale_x = scaled_image.width() / w
                scale_y = scaled_image.height() / h

                if self.show_mesh:
                    painter.setPen(QPen(Qt.blue, 1))
                    for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                        start = self.landmarks[connection[0]]
                        end = self.landmarks[connection[1]]
                        x1 = int(start.x * w * scale_x) + x_offset
                        y1 = int(start.y * h * scale_y) + y_offset
                        x2 = int(end.x * w * scale_x) + x_offset
                        y2 = int(end.y * h * scale_y) + y_offset
                        painter.drawLine(x1, y1, x2, y2)

                if self.show_landmarks:
                    painter.setPen(QPen(Qt.green, 3))
                    for landmark in self.landmarks:
                        x = int(landmark.x * w * scale_x) + x_offset
                        y = int(landmark.y * h * scale_y) + y_offset
                        painter.drawPoint(x, y)

            painter.endNativePainting()

            # Render 3D model if available
            if False and (hasattr(self, 'vao') and self.vao is not None and
                    hasattr(self, 'model_faces') and self.model_faces is not None and
                    self.landmarks is not None and self.shader_program):

                glUseProgram(self.shader_program)

                # Create model matrix
                model_matrix = np.eye(4, dtype=np.float32)

                # Get nose position for model placement
                nose = self.landmarks[1]  # nose tip
                x = (nose.x * 2.0 - 1.0)
                y = -(nose.y * 2.0 - 1.0)
                z = -2.0  # Fixed distance from camera

                # Update model matrix
                model_matrix[0:3, 3] = [x, y, z]  # Translation
                model_matrix[0:3, 0:3] *= self.model_scale  # Scale

                # Set uniforms
                glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "model"),
                                   1, GL_FALSE, model_matrix)
                glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "view"),
                                   1, GL_FALSE, self.view)
                glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"),
                                   1, GL_FALSE, self.projection)

                try:
                    # Draw model
                    glBindVertexArray(self.vao)
                    num_indices = len(self.model_faces.flatten())
                    glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
                    glBindVertexArray(0)
                except Exception as e:
                    print(f"Error rendering model: {e}")

                glUseProgram(0)

            painter.end()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def update_camera_image(self, image):
        self.image = image.copy()
        self.update()

    def update_model_position(self, landmarks):
        """Update face landmarks and repaint. landmarks can be None if no face is detected."""
        self.landmarks = landmarks  # Will be None if no face is detected
        self.update()

    def load_model(self, model_path):
        """Load and prepare 3D model for rendering"""
        try:
            # Load model using trimesh
            self.model = trimesh.load(model_path)

            # Get vertices and faces
            self.model_vertices = self.model.vertices.astype(np.float32)
            self.model_faces = self.model.faces.astype(np.uint32)

            # Create VAO
            self.vao = glGenVertexArrays(1)
            glBindVertexArray(self.vao)

            # Create and fill VBO with vertices
            self.vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, self.model_vertices.nbytes,
                         self.model_vertices, GL_STATIC_DRAW)

            # Create and fill EBO with faces
            self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.model_faces.nbytes,
                         self.model_faces, GL_STATIC_DRAW)

            # Set vertex attributes
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)

            # Unbind VAO
            glBindVertexArray(0)

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

