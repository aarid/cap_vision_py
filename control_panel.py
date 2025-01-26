from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QPushButton,
                             QSlider, QLabel, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal


class ControlPanel(QGroupBox):
    # Define signals
    mesh_visibility_changed = pyqtSignal(bool)
    landmarks_visibility_changed = pyqtSignal(bool)
    scale_changed = pyqtSignal(float)
    position_changed = pyqtSignal(str, float)  # axis, value
    cap_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Cap selection
        self.setup_cap_selection(layout)

        # Visualization controls
        self.setup_visualization(layout)

        # Adjustment controls
        self.setup_adjustments(layout)

        # Add stretch to push controls to the top
        layout.addStretch()

    def setup_cap_selection(self, parent_layout):
        cap_group = QGroupBox("Cap Selection")
        layout = QVBoxLayout(cap_group)

        self.cap_combo = QComboBox()
        self.cap_combo.addItems(["Cap 1", "Cap 2", "Cap 3"])
        self.cap_combo.currentIndexChanged.connect(self.cap_changed.emit)
        layout.addWidget(self.cap_combo)

        parent_layout.addWidget(cap_group)

    def setup_visualization(self, parent_layout):
        viz_group = QGroupBox("Visualization")
        layout = QVBoxLayout(viz_group)

        self.show_mesh_btn = QPushButton("Show Face Mesh")
        self.show_mesh_btn.setCheckable(True)
        self.show_mesh_btn.setChecked(True)
        self.show_mesh_btn.toggled.connect(self.mesh_visibility_changed.emit)
        layout.addWidget(self.show_mesh_btn)

        self.show_landmarks_btn = QPushButton("Show Landmarks")
        self.show_landmarks_btn.setCheckable(True)
        self.show_landmarks_btn.toggled.connect(self.landmarks_visibility_changed.emit)
        layout.addWidget(self.show_landmarks_btn)

        parent_layout.addWidget(viz_group)

    def setup_adjustments(self, parent_layout):
        adjust_group = QGroupBox("Adjustments")
        layout = QVBoxLayout(adjust_group)

        # Scale slider
        scale_layout = QVBoxLayout()
        scale_label = QLabel("Scale:")
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(50, 150)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(
            lambda v: self.scale_changed.emit(v / 100.0))
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_slider)
        layout.addLayout(scale_layout)

        # Position sliders
        for axis in ['X', 'Y', 'Z']:
            pos_layout = QVBoxLayout()
            pos_label = QLabel(f"{axis} Position:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)
            slider.valueChanged.connect(
                lambda v, a=axis: self.position_changed.emit(a, v / 100.0))
            setattr(self, f'pos_{axis.lower()}_slider', slider)
            pos_layout.addWidget(pos_label)
            pos_layout.addWidget(slider)
            layout.addLayout(pos_layout)

        parent_layout.addWidget(adjust_group)