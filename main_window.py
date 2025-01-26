from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
import cv2
import mediapipe as mp
from gl_widget import GLWidget
from control_panel import ControlPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CapVision")

        # MediaPipe setup
        self.setup_mediapipe()

        # Camera setup
        self.cap = cv2.VideoCapture(0)

        # UI setup
        self.setup_ui()

        # Timer setup
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Window setup
        self.resize(1200, 800)

    def setup_mediapipe(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # GL Widget setup
        self.gl_widget = GLWidget()
        self.gl_widget.setMinimumSize(800, 600)
        layout.addWidget(self.gl_widget, stretch=2)

        # Control Panel setup
        self.control_panel = ControlPanel()
        self.setup_control_connections()
        layout.addWidget(self.control_panel, stretch=1)

        # Load model after GL context is created
        QTimer.singleShot(100, self.load_model)  # Delay model loading

    def load_model(self):
        try:
            success = self.gl_widget.load_model("10131_BaseballCap_v2_L3.obj")
            print(f"Model loading {'successful' if success else 'failed'}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def setup_control_connections(self):
        # Connect control panel signals
        self.control_panel.mesh_visibility_changed.connect(
            self.gl_widget.set_show_mesh)
        self.control_panel.landmarks_visibility_changed.connect(
            self.gl_widget.set_show_landmarks)
        self.control_panel.scale_changed.connect(
            self.gl_widget.set_scale)
        self.control_panel.position_changed.connect(
            self.gl_widget.set_position)
        self.control_panel.cap_changed.connect(
            self.gl_widget.set_cap_model)

    def update_frame(self):
        success, image = self.cap.read()
        if not success:
            return

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True

        # Update OpenGL widget
        self.gl_widget.update_camera_image(image)

        # Clear landmarks if no face is detected
        if not results.multi_face_landmarks:
            self.gl_widget.update_model_position(None)  # Pass None when no face is detected
        else:
            self.gl_widget.update_model_position(results.multi_face_landmarks[0].landmark)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()