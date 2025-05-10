import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStatusBar
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject # QObject added for state management
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
from collections import deque

# --- OpenCV Configuration (from previous script) ---
MIN_MATCH_COUNT = 10
LOWE_RATIO_TEST = 0.75
KEY_TO_CYCLE_QT = Qt.Key_N
KEY_TO_QUIT_QT = Qt.Key_Q

# --- Chart Configuration ---
MAX_CHART_POINTS = 100  # Number of data points to display on the chart

# --- State Management Object ---
class AppState(QObject):
    state_changed = pyqtSignal(int)
    capture_reference_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    STATE_WAITING_FOR_REFERENCE = 0
    STATE_TRACKING = 1

    def __init__(self):
        super().__init__()
        self._current_state = self.STATE_WAITING_FOR_REFERENCE

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, value):
        if self._current_state != value:
            self._current_state = value
            self.state_changed.emit(value)

    def request_capture_reference(self):
        self.capture_reference_requested.emit()

    def request_reset(self):
        self.reset_requested.emit()

app_state = AppState() # Global instance for simplicity, or pass around

# --- OpenCV Processing Thread ---
class OpenCVThread(QThread):
    frame_ready = pyqtSignal(QImage)
    matches_count_ready = pyqtSignal(int)
    status_message = pyqtSignal(str)

    def __init__(self, app_state_ref):
        super().__init__()
        self.running = False
        self.app_state = app_state_ref # Reference to the shared state object

        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self.orb = None
        self.bf_matcher = None

        # Connect to state signals
        self.app_state.capture_reference_requested.connect(self.prepare_for_reference_capture)
        self.app_state.reset_requested.connect(self.reset_reference)


    def initialize_features(self):
        self.orb = cv2.ORB_create(nfeatures=1000) # Adjusted for potentially faster processing
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def prepare_for_reference_capture(self):
        self._capture_next_frame_as_reference = True

    def reset_reference(self):
        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self._capture_next_frame_as_reference = False # Ensure this is reset too
        self.app_state.current_state = AppState.STATE_WAITING_FOR_REFERENCE
        self.status_message.emit("Reference reset. Aim and press 'N' to capture.")


    def run(self):
        self.running = True
        self.initialize_features()
        self._capture_next_frame_as_reference = False

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_message.emit("Error: Cannot open camera.")
            self.running = False
            return

        if self.app_state.current_state == AppState.STATE_WAITING_FOR_REFERENCE:
             self.status_message.emit("Aim camera and press 'N' to capture reference.")


        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_message.emit("Error: Can't receive frame.")
                time.sleep(0.5) # Avoid busy-looping on error
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image.copy()) # Emit a copy

            num_good_matches = 0

            if self._capture_next_frame_as_reference:
                self.reference_frame = frame.copy() # Use BGR frame for OpenCV processing
                self.reference_kp, self.reference_des = self.orb.detectAndCompute(self.reference_frame, None)
                if self.reference_des is None or len(self.reference_kp) < MIN_MATCH_COUNT:
                    self.status_message.emit("Not enough features in reference. Try again.")
                    self.reference_frame = None # Invalidate
                    self.app_state.current_state = AppState.STATE_WAITING_FOR_REFERENCE # Stay in waiting
                else:
                    self.status_message.emit(f"Reference captured ({len(self.reference_kp)} pts). Tracking...")
                    self.app_state.current_state = AppState.STATE_TRACKING
                self._capture_next_frame_as_reference = False


            if self.app_state.current_state == AppState.STATE_TRACKING and self.reference_frame is not None:
                current_kp, current_des = self.orb.detectAndCompute(frame, None) # Use BGR frame

                if current_des is not None and len(current_des) > 0 and self.reference_des is not None:
                    all_matches = self.bf_matcher.knnMatch(self.reference_des, current_des, k=2)
                    good_matches = []
                    for m_arr in all_matches:
                        if len(m_arr) == 2:
                            m, n = m_arr
                            if m.distance < LOWE_RATIO_TEST * n.distance:
                                good_matches.append(m)
                    num_good_matches = len(good_matches)

                    if num_good_matches >= MIN_MATCH_COUNT:
                        src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        # We won't draw on the frame here, as the QImage is already emitted.
                        # Drawing would need to be done before conversion or by sending drawing info separately.
                        # For simplicity, we'll just emit the match count.
                        if H is None:
                             num_good_matches = -1 # Indicate homography failure despite matches
                    # else: not enough good matches, num_good_matches already reflects this
                # else: no current descriptors or no reference descriptors
            # else: not tracking or no reference frame

            self.matches_count_ready.emit(num_good_matches)
            self.msleep(30) # Control frame rate slightly, adjust as needed

        cap.release()
        self.status_message.emit("Camera released.")

    def stop(self):
        self.running = False
        self.wait()


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self, app_state_ref):
        super().__init__()
        self.app_state = app_state_ref
        self.setWindowTitle("OpenCV Homography with Qt5 Chart")
        self.setGeometry(100, 100, 1000, 600) # x, y, width, height

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget) # Main layout: Video | Chart + Controls

        # Video display
        video_layout = QVBoxLayout()
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setFixedSize(640, 480) # Adjust as needed
        self.video_label.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout, 2) # Video takes 2/3 of space

        # Chart and controls layout
        controls_chart_layout = QVBoxLayout()

        # Chart
        self.match_chart_widget = pg.PlotWidget()
        self.match_chart_widget.setBackground('w')
        self.match_chart_widget.setTitle("Good Feature Matches", color="k", size="12pt")
        self.match_chart_widget.setLabel('left', 'Match Count', color='k')
        self.match_chart_widget.setLabel('bottom', 'Time (frames)', color='k')
        self.match_chart_widget.showGrid(x=True, y=True)
        self.match_data_line = self.match_chart_widget.plot(pen=pg.mkPen('b', width=2))
        self.match_history = deque(maxlen=MAX_CHART_POINTS)
        self.time_points = deque(maxlen=MAX_CHART_POINTS) # X-axis data
        self.current_time_step = 0

        controls_chart_layout.addWidget(self.match_chart_widget)
        main_layout.addLayout(controls_chart_layout, 1) # Chart/Controls take 1/3

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # OpenCV Thread
        self.opencv_thread = OpenCVThread(self.app_state)
        self.opencv_thread.frame_ready.connect(self.update_video_frame)
        self.opencv_thread.matches_count_ready.connect(self.update_matches_chart)
        self.opencv_thread.status_message.connect(self.show_status_message)
        self.app_state.state_changed.connect(self.on_state_changed_gui) # Update GUI based on state

        self.opencv_thread.start()

        self.show_status_message("Application started. Press 'N' to capture reference.")
        self.on_state_changed_gui(self.app_state.current_state) # Initial status message based on state


    def update_video_frame(self, q_image):
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_matches_chart(self, count):
        self.match_history.append(count if count >= 0 else 0) # Plot 0 if homography failed (-1)
        self.time_points.append(self.current_time_step)
        self.current_time_step += 1

        # Prune data if necessary (deque handles maxlen, but ensure x-axis matches)
        if len(self.time_points) > MAX_CHART_POINTS:
            self.time_points.popleft() # Should be handled by deque's maxlen automatically for y too

        self.match_data_line.setData(list(self.time_points), list(self.match_history))

    def show_status_message(self, message):
        self.status_bar.showMessage(message)
        print(message) # Also print to console for debugging

    def on_state_changed_gui(self, state):
        if state == AppState.STATE_WAITING_FOR_REFERENCE:
            self.show_status_message("STATE: Waiting for Reference. Aim and press 'N'.")
        elif state == AppState.STATE_TRACKING:
            self.show_status_message("STATE: Tracking. Press 'N' to reset reference.")


    def keyPressEvent(self, event):
        key = event.key()
        if key == KEY_TO_QUIT_QT:
            self.close()
        elif key == KEY_TO_CYCLE_QT:
            if self.app_state.current_state == AppState.STATE_WAITING_FOR_REFERENCE:
                self.show_status_message("Attempting to capture reference frame...")
                self.app_state.request_capture_reference() # Signal the thread
            elif self.app_state.current_state == AppState.STATE_TRACKING:
                self.show_status_message("Resetting reference...")
                self.app_state.request_reset() # Signal the thread
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.show_status_message("Closing application...")
        self.opencv_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # pg.setConfigOptions(antialias=True) # Optional: for smoother lines in chart
    main_window = MainWindow(app_state)
    main_window.show()
    sys.exit(app.exec_())
