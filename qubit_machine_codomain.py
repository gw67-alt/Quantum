import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStatusBar
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont
import pyqtgraph as pg
from collections import deque
import os

# Adjustable game parameters
# Initial threshold - will be dynamically updated to use the average match count
MATCH_THRESHOLD_FOR_GUESS = 0.5 # Initial value, will be adjusted dynamically
STARTING_CREDITS = 100
COST_PER_GUESS = 1
WIN_CREDITS = 150

# Game state
game_state = {
    "credits": STARTING_CREDITS,
    "wins": 0,
    "losses": 0
}

# Load data file safely
data = []
try:
    with open("x.txt", 'r', encoding='utf-8') as file:
        data = [line.strip() for line in file.readlines() if line.strip()]
    if not data:
        print("Warning: x.txt file is empty. Using dummy data.")
        data = ["55", "55", "AA", "55", "BB"]  # Dummy data if file is empty
except FileNotFoundError:
    print("Warning: x.txt file not found. Creating with dummy data.")
    with open("x.txt", 'w', encoding='utf-8') as file:
        file.write("55\n55\nAA\n55\nBB\n")
    data = ["55", "55", "AA", "55", "BB"]  # Dummy data for new file

# --- OpenCV Configuration ---
MIN_MATCH_COUNT = 5  # Lowered from 10 to be more lenient
LOWE_RATIO_TEST = 0.7  # Increased from 0.10 to be less strict
KEY_TO_CYCLE_QT = Qt.Key_N
KEY_TO_QUIT_QT = Qt.Key_Q

# --- Chart Configuration ---
MAX_CHART_POINTS = 100  # Number of data points to display on the chart
MOVING_AVG_WINDOW = 150  # Window size for the moving average - reduced for quicker response

# --- Guessing Configuration ---
GUESS_TRIGGER_COUNT = 8  # Number of samples before attempting a guess

# --- State Management Object ---
class AppState(QObject):
    state_changed = pyqtSignal(int)
    capture_reference_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    game_state_updated = pyqtSignal(dict)  # New signal for game state updates

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
    
    def update_game_state(self, state_dict):
        self.game_state_updated.emit(state_dict)

app_state = AppState()

# --- OpenCV Processing Thread ---
class OpenCVThread(QThread):
    frame_ready = pyqtSignal(QImage)
    matches_count_ready = pyqtSignal(int)
    status_message = pyqtSignal(str)

    def __init__(self, app_state_ref):
        super().__init__()
        self.running = False
        self.app_state = app_state_ref

        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self.orb = None
        self.bf_matcher = None
        self._capture_next_frame_as_reference = False

        self.app_state.capture_reference_requested.connect(self.prepare_for_reference_capture)
        self.app_state.reset_requested.connect(self.reset_reference)

    def initialize_features(self):
        # Enhanced ORB parameters for better feature detection
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        # BFMatcher with crossCheck=False is used for knnMatch
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def prepare_for_reference_capture(self):
        self._capture_next_frame_as_reference = True

    def reset_reference(self):
        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self._capture_next_frame_as_reference = False
        # This will trigger state_changed signal handled by MainWindow
        self.app_state.current_state = AppState.STATE_WAITING_FOR_REFERENCE
        
    def preprocess_frame(self, frame):
        """Enhance frame for better feature detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Optional: Apply slight Gaussian blur to reduce noise
        # enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced

    def run(self):
        self.running = True
        self.initialize_features()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_message.emit("Error: Cannot open camera.")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_message.emit("Error: Can't receive frame.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image.copy())  # Emit a copy

            num_good_matches_for_signal = 0  # Default to 0 (no good matches / not tracking)

            if self._capture_next_frame_as_reference:
                self.reference_frame = frame.copy()  # Keep original for display
                processed_ref = self.preprocess_frame(self.reference_frame)
                self.reference_kp, self.reference_des = self.orb.detectAndCompute(processed_ref, None)

                if self.reference_des is None or len(self.reference_kp) < MIN_MATCH_COUNT:
                    self.status_message.emit(f"Ref. Capture Failed: Not enough features ({len(self.reference_kp) if self.reference_kp is not None else 0}). Try again.")
                    self.reference_frame = None  # Clear invalid reference
                    self.reference_kp = None
                    self.reference_des = None
                    self.app_state.current_state = AppState.STATE_WAITING_FOR_REFERENCE
                else:
                    self.status_message.emit(f"Reference Captured ({len(self.reference_kp)} keypoints). Tracking...")
                    self.app_state.current_state = AppState.STATE_TRACKING
                self._capture_next_frame_as_reference = False
                self.matches_count_ready.emit(0)  # Emit 0 matches right after capture attempt


            elif self.app_state.current_state == AppState.STATE_TRACKING and self.reference_frame is not None and self.reference_des is not None:
                processed_frame = self.preprocess_frame(frame)
                current_kp, current_des = self.orb.detectAndCompute(processed_frame, None)
                actual_good_matches_count = 0

                if current_des is not None and len(current_des) > 0:
                    try:
                        # Try to perform knnMatch
                        all_matches = self.bf_matcher.knnMatch(self.reference_des, current_des, k=2)
                        good_matches = []

                        for m_arr in all_matches:
                            if len(m_arr) == 2:
                                m, n = m_arr
                                if m.distance < LOWE_RATIO_TEST * n.distance:
                                    good_matches.append(m)

                        actual_good_matches_count = len(good_matches)

                        # Improved logging
                        if actual_good_matches_count < MIN_MATCH_COUNT:
                            self.status_message.emit(f"Low matches: {actual_good_matches_count}/{MIN_MATCH_COUNT}")
                        
                        if actual_good_matches_count >= MIN_MATCH_COUNT:
                            try:
                                src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                                # RANSAC threshold increased slightly for more tolerance
                                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

                                if H is None:
                                    num_good_matches_for_signal = actual_good_matches_count // 2  # Half the matches instead of -1
                                    self.status_message.emit("Homography calculation failed")
                                else:
                                    num_good_matches_for_signal = actual_good_matches_count
                            except Exception as e:
                                self.status_message.emit(f"Homography error: {str(e)}")
                                num_good_matches_for_signal = 0
                        else:
                            num_good_matches_for_signal = actual_good_matches_count
                    except Exception as e:
                        self.status_message.emit(f"Match error: {str(e)}")
                        num_good_matches_for_signal = 0
                else:
                    num_good_matches_for_signal = 0
                    self.status_message.emit("No features detected in current frame")

                self.matches_count_ready.emit(num_good_matches_for_signal)

            else:  # Waiting for reference or reference invalid
                self.matches_count_ready.emit(0)  # Emit 0 if not tracking

        cap.release()
        self.status_message.emit("Camera released.")

    def stop(self):
        self.running = False
        self.wait()


# --- Main Application Window (Updated with Guessing Logic) ---
class MainWindow(QMainWindow):
    def __init__(self, app_state_ref):
        super().__init__()
        self.app_state = app_state_ref
        self.setWindowTitle("OpenCV Homography Tracker with Guessing Game")
        self.setGeometry(100, 100, 1200, 700)

        # Dynamic threshold variables
        self.current_threshold = MATCH_THRESHOLD_FOR_GUESS
        self.threshold_history = deque(maxlen=30)  # Store recent match counts for dynamic threshold

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: video and game stats
        left_layout = QVBoxLayout()
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)
        
        # Game stats panel
        stats_layout = QHBoxLayout()
        
        self.credits_label = QLabel(f"Credits: {game_state['credits']}")
        self.credits_label.setFont(QFont('Arial', 14))
        self.credits_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.wins_label = QLabel(f"Wins: {game_state.get('wins', 0)}")
        self.wins_label.setFont(QFont('Arial', 14))
        self.wins_label.setStyleSheet("color: blue;")
        
        self.losses_label = QLabel(f"Losses: {game_state.get('losses', 0)}")
        self.losses_label.setFont(QFont('Arial', 14))
        self.losses_label.setStyleSheet("color: red;")
        
        stats_layout.addWidget(self.credits_label)
        stats_layout.addWidget(self.wins_label)
        stats_layout.addWidget(self.losses_label)
        
        left_layout.addLayout(stats_layout)
        main_layout.addLayout(left_layout, 2)

        # Right side: chart and controls
        controls_chart_layout = QVBoxLayout()
        self.match_chart_widget = pg.PlotWidget()
        self.match_chart_widget.setBackground('w')
        self.match_chart_widget.setTitle("Feature Matches Analysis", color="k", size="12pt")
        self.match_chart_widget.setLabel('left', 'Match Count', color='k')
        self.match_chart_widget.setLabel('bottom', 'Time (frames)', color='k')
        self.match_chart_widget.showGrid(x=True, y=True)
        self.match_chart_widget.addLegend()

        # Add threshold line for guessing
        self.threshold_line = pg.InfiniteLine(pos=self.current_threshold, angle=0, 
                                            pen=pg.mkPen('g', width=2, style=Qt.DashLine),
                                            label=f'Guess Threshold ({self.current_threshold:.2f})')
        self.match_chart_widget.addItem(self.threshold_line)

        self.raw_match_data_line = self.match_chart_widget.plot(pen=pg.mkPen('b', width=2), name="Raw Matches")
        self.raw_match_history = deque(maxlen=MAX_CHART_POINTS)

        self.avg_match_data_line = self.match_chart_widget.plot(pen=pg.mkPen('r', width=2, style=Qt.DashLine), name=f"Avg Matches (Win: {MOVING_AVG_WINDOW})")
        self.avg_match_history = deque(maxlen=MAX_CHART_POINTS)

        self.time_points = deque(maxlen=MAX_CHART_POINTS)
        self.current_time_step = 0

        self.guess_trigger_sample_counter = 0

        controls_chart_layout.addWidget(self.match_chart_widget)
        
        # Add instructions
        instructions_label = QLabel("Instructions: Press 'N' to capture reference or reset. Press 'Q' to quit.")
        instructions_label.setAlignment(Qt.AlignCenter)
        instructions_label.setFont(QFont('Arial', 10))
        controls_chart_layout.addWidget(instructions_label)
        
        main_layout.addLayout(controls_chart_layout, 3)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.opencv_thread = OpenCVThread(self.app_state)
        self.opencv_thread.frame_ready.connect(self.update_video_frame)
        self.opencv_thread.matches_count_ready.connect(self.update_matches_chart_and_guess)
        self.opencv_thread.status_message.connect(self.show_status_message)
        self.app_state.state_changed.connect(self.on_state_changed_gui)
        self.app_state.game_state_updated.connect(self.update_game_stats)
        
        # Game state tracking variables
        self.data_index = 0
        self.ready_count = 0
        self.success_count = 0
        self.init_count = 0
        # Start the thread
        self.opencv_thread.start()
        self.on_state_changed_gui(self.app_state.current_state)

    def update_video_frame(self, q_image):
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_matches_chart_and_guess(self, raw_match_count_from_thread):
        actual_plot_count = raw_match_count_from_thread if raw_match_count_from_thread >= 0 else 0

        self.raw_match_history.append(actual_plot_count)
        self.time_points.append(self.current_time_step)
        self.current_time_step += 1

        # Update threshold history and recalculate dynamic threshold
        self.threshold_history.append(actual_plot_count)
        
        # Calculate moving average for chart display
        current_avg = 0.0
        if len(self.raw_match_history) > 0:
            avg_window_data = list(self.raw_match_history)[-MOVING_AVG_WINDOW:]
            if avg_window_data:  # Ensure not empty
                current_avg = np.mean(avg_window_data)
        self.avg_match_history.append(current_avg)
        
        # Update dynamic threshold based on recent match history
        if len(self.threshold_history) > 5:  # Need at least a few data points
            self.current_threshold = np.mean(self.threshold_history)
            # Update the threshold line position and label
            self.threshold_line.setValue(self.current_threshold)
            self.threshold_line.label.setText(f'Guess Threshold ({self.current_threshold:.2f})')

        list_time_points = list(self.time_points)
        list_raw_history = list(self.raw_match_history)
        list_avg_history = list(self.avg_match_history)

        self.raw_match_data_line.setData(list_time_points, list_raw_history)
        self.avg_match_data_line.setData(list_time_points, list_avg_history)

        # Only process guesses when actively tracking
        if self.app_state.current_state == AppState.STATE_TRACKING:
            self.guess_trigger_sample_counter += 1
            
            # Only make a guess every GUESS_TRIGGER_COUNT frames
            if self.guess_trigger_sample_counter >= GUESS_TRIGGER_COUNT:
                self.process_guess(actual_plot_count)
                self.guess_trigger_sample_counter = 0

    def process_guess(self, match_count):
        # Check if the match count is below threshold
        is_below = match_count < self.current_threshold

        if is_below and game_state["credits"] > 0:
            self.ready_count += 1
            
            # Guard against empty data list
            if not data:
                self.show_status_message("Error: No data available for guessing!", 3000)
                return
                
            try:
                # Try to interpret the current data value as hex
                current_value = data[self.init_count].strip()
                hex_value = int(current_value, 16)
                
                if hex_value == 0x55:  # Win condition 
                    self.success_count += 1
                    game_state["credits"] += COST_PER_GUESS
                    game_state["wins"] = game_state.get("wins", 0) + 1
                    self.show_status_message(f"Win! {current_value} = 0x55. +{WIN_CREDITS} credits! (Threshold: {self.current_threshold:.2f})", 2000)
                else:
                    game_state["credits"] -= COST_PER_GUESS
                    game_state["losses"] = game_state.get("losses", 0) + 1
                    self.show_status_message(f"Lost! {current_value} ≠ 0x55. -{COST_PER_GUESS} credits. (Threshold: {self.current_threshold:.2f})", 2000)
                    
                # Log debug info
                print(f"Guessed 'Below': {current_value} @ init {self.init_count}, " 
                      f"Ready: {self.ready_count}, Success: {self.success_count}, "
                      f"Credits: {game_state['credits']}, Threshold: {self.current_threshold:.2f}")
                      
            except (ValueError, IndexError) as e:
                print(f"Error processing data at index {self.data_index}: {e}")
                self.show_status_message(f"Data processing error: {str(e)}", 3000)
                
            finally:
                # Move to next data point, wrap around if needed
                False
        else:
            # Guessed "high" - always a loss
            game_state["losses"] = game_state.get("losses", 0) + 1
            self.show_status_message(f"Guessed High: Lost {COST_PER_GUESS} credits. (Threshold: {self.current_threshold:.2f})", 2000)
            print(f"Guessed 'High', Lost {COST_PER_GUESS} Credits. Total: {game_state['credits']}, Threshold: {self.current_threshold:.2f}")
            
        # Update UI with new game state
        self.app_state.update_game_state(game_state)
        self.init_count = (self.init_count + 1) % len(data)

    def update_game_stats(self, state_dict):
        """Update the game statistics display"""
        self.credits_label.setText(f"Credits: {state_dict['credits']}")
        self.wins_label.setText(f"Wins: {state_dict.get('wins', 0)}")
        self.losses_label.setText(f"Losses: {state_dict.get('losses', 0)}")

    def show_status_message(self, message, timeout=0):
        self.status_bar.showMessage(message, timeout)

    def on_state_changed_gui(self, state):
        self.guess_trigger_sample_counter = 0
        if state == AppState.STATE_WAITING_FOR_REFERENCE:
            self.show_status_message("STATE: Waiting for Reference. Aim and press 'N'.")
            self.clear_chart_data()
        elif state == AppState.STATE_TRACKING:
            self.show_status_message("STATE: Tracking active. Press 'N' to reset.")

    def clear_chart_data(self):
        self.raw_match_history.clear()
        self.avg_match_history.clear()
        self.time_points.clear()
        self.threshold_history.clear()  # Also clear threshold history when resetting
        self.current_time_step = 0
        # Reset threshold to initial value
        self.current_threshold = MATCH_THRESHOLD_FOR_GUESS
        self.threshold_line.setValue(self.current_threshold)
        self.threshold_line.label.setText(f'Guess Threshold ({self.current_threshold:.2f})')
        self.raw_match_data_line.setData([], [])
        self.avg_match_data_line.setData([], [])

    def keyPressEvent(self, event):
        key = event.key()
        if key == KEY_TO_QUIT_QT:
            self.close()
        elif key == KEY_TO_CYCLE_QT:
            # Temporary messages for key presses, main state message will be set by on_state_changed_gui
            if self.app_state.current_state == AppState.STATE_WAITING_FOR_REFERENCE:
                self.show_status_message("GUI: Requesting reference capture...", 2000)
                self.app_state.request_capture_reference()
            elif self.app_state.current_state == AppState.STATE_TRACKING:
                self.show_status_message("GUI: Requesting reset...", 2000)
                self.app_state.request_reset()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.show_status_message("Closing application...")
        self.opencv_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    main_window = MainWindow(app_state)
    main_window.show()
    sys.exit(app.exec_())
