# Full code with modifications for RMS deviation metric and ROI functionality
import cv2
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from threading import Lock # Using Lock
import matplotlib

# --- Try using Qt5Agg backend first ---
try:
    matplotlib.use('Qt5Agg')
    print("Using Matplotlib backend: Qt5Agg")
except ImportError:
    print("Warning: Qt5Agg backend not found or PyQt5 not installed (pip install PyQt5).")
    try:
        # Fallback to TkAgg
        matplotlib.use('TkAgg')
        print("Using Matplotlib backend: TkAgg")
    except ImportError:
        print("Warning: TkAgg backend not found either. Using default Matplotlib backend.")

# Set parameters after setting the backend
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.autolayout'] = True
plt.rcParams['lines.linewidth'] = 2

class DataPlotter:
    def __init__(self, max_points=100):
        """
        Initialize a real-time data plotter using matplotlib.
        Plots Straight Line Count, Average RMS Deviation, and Straight/Total Ratio.
        Parameters: max_points (int): Max history length.
        """
        self.max_points = max_points
        self.lock = Lock()
        self.timestamps = deque(maxlen=max_points)
        self.straight_lines = deque(maxlen=max_points)
        self.avg_deviations = deque(maxlen=max_points) # Deque for average RMS deviation
        self.line_ratios = deque(maxlen=max_points)

        # Initial dummy data point
        self.timestamps.append(0)
        self.straight_lines.append(0)
        self.avg_deviations.append(0.0)
        self.line_ratios.append(0)

        plt.style.use('dark_background')
        # Keep 3 subplots: Straight Count, Avg RMS Deviation, Ratio
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        self.fig.suptitle('Camera Stream Analysis (ROI)', fontsize=16)

        # --- Plot Definitions ---
        # Axes 0: Straight Lines
        self.straight_line_plot, = self.axes[0].plot([], [], 'g-', label='Straight Lines (ROI)')
        self.axes[0].set_title('Straight Line Count (ROI)')
        self.axes[0].set_ylabel('Count')
        self.axes[0].legend(loc='upper left')
        self.axes[0].grid(True, linestyle='--', alpha=0.6)

        # Axes 1: Average RMS Deviation
        self.deviation_plot, = self.axes[1].plot([], [], 'r-', label='RMS Deviation (ROI)') # CHANGED label
        self.axes[1].set_title('RMS Deviation (ROI)') # CHANGED title
        self.axes[1].set_ylabel('Deviation (pixels)') # CHANGED ylabel
        self.axes[1].set_ylim(0, 5.0) # Initial guess for pixel deviation, will be overridden dynamically
        self.axes[1].legend(loc='upper left')
        self.axes[1].grid(True, linestyle='--', alpha=0.6)

        # Axes 2: Line Ratio
        self.ratio_plot, = self.axes[2].plot([], [], 'b-', label='Straight/Total Ratio (ROI)')
        self.axes[2].set_title('Line Ratio (Straight / Total) (ROI)')
        self.axes[2].set_ylabel('Ratio')
        self.axes[2].set_ylim(0, 1.05)
        self.axes[2].legend(loc='upper left')
        self.axes[2].grid(True, linestyle='--', alpha=0.6)
        self.axes[2].set_xlabel('Time (s)') # Add xlabel to the last plot

        # Initial Axis Limits
        for ax in self.axes:
            ax.set_xlim(0, 10)
        self.axes[0].set_ylim(0, 10) # For straight count

        self.ani = FuncAnimation(
            self.fig, self.update_plot, interval=100,
            blit=False, cache_frame_data=False
        )

        if not plt.isinteractive(): plt.ion()
        plt.show(block=False)
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
        self.start_time = time.time()

    def add_data_point(self, straight_lines, avg_deviation, line_ratio): # CHANGED signature
        """Add a new data point to the plot queues."""
        with self.lock:
            current_time = time.time() - self.start_time
            if not self.timestamps or current_time > self.timestamps[-1] + 1e-6:
                self.timestamps.append(current_time)
                self.straight_lines.append(straight_lines)
                self.avg_deviations.append(avg_deviation) # CHANGED
                self.line_ratios.append(line_ratio)
            elif self.timestamps: # Update last point if timestamp hasn't changed significantly
                self.straight_lines[-1] = straight_lines
                self.avg_deviations[-1] = avg_deviation # CHANGED
                self.line_ratios[-1] = line_ratio


    def update_plot(self, frame):
        """Update the plot with new data. Called by FuncAnimation."""
        with self.lock:
            # Handle the initial dummy point correctly
            if len(self.timestamps) <= 1 and self.timestamps[0] == 0:
                # Return existing artists if no real data yet
                return self.straight_line_plot, self.deviation_plot, self.ratio_plot # CHANGED

            # Convert deques to lists for plotting
            times_list = list(self.timestamps)
            straight_list = list(self.straight_lines)
            deviation_list = list(self.avg_deviations) # CHANGED
            ratio_list = list(self.line_ratios)

            # Skip the initial dummy point (index 0) if more data exists
            if len(times_list) > 1:
                times_plot = times_list[1:]
                straight_plot = straight_list[1:]
                deviation_plot_data = deviation_list[1:] # CHANGED
                ratio_plot_data = ratio_list[1:]
            else: # Only the dummy point exists, nothing to plot yet
                return self.straight_line_plot, self.deviation_plot, self.ratio_plot # CHANGED

        # Check if there's actual data to plot after slicing
        if not times_plot:
            return self.straight_line_plot, self.deviation_plot, self.ratio_plot # CHANGED

        # --- Set Plot Data ---
        self.straight_line_plot.set_data(times_plot, straight_plot)
        self.deviation_plot.set_data(times_plot, deviation_plot_data) # CHANGED
        self.ratio_plot.set_data(times_plot, ratio_plot_data)

        # --- Dynamic X-axis scaling (sliding window) ---
        history_duration = 20.0 # Show last 20 seconds
        current_max_time = times_plot[-1] if times_plot else 0
        x_min = max(0, current_max_time - history_duration)
        x_max = max(x_min + 10.0, current_max_time + 2.0) # Ensure x_max is always ahead, provide buffer

        # --- Dynamic Y-axis scaling based on visible data ---
        visible_indices = [i for i, t in enumerate(times_plot) if t >= x_min]
        if not visible_indices:
            y_max_straight = 10 # Default if no data in view
            y_max_deviation = 2.0 # Default for deviation plot (pixels)
        else:
            first_visible_index = visible_indices[0]
            # Slice data lists to get only the visible points for max calculation
            visible_straight = straight_plot[first_visible_index:]
            visible_deviation = deviation_plot_data[first_visible_index:] # CHANGED

            # Use default values in max() to handle cases where slices might become empty
            max_s = max(visible_straight, default=0)
            max_dev = max(visible_deviation, default=0.0) # CHANGED

            # Ensure axes don't shrink below a minimum value
            y_max_straight = max(max_s, 10)
            y_max_deviation = max(max_dev, 1.0) # Min upper Y limit for deviation (e.g., 1 pixel) # CHANGED


        # Update axes limits
        for ax in self.axes:
             ax.set_xlim(x_min, x_max)
        # Add padding to y-limits (e.g., 20%)
        self.axes[0].set_ylim(0, y_max_straight * 1.2) # Straight count
        self.axes[1].set_ylim(0, y_max_deviation * 1.2) # Average RMS deviation # CHANGED
        self.axes[2].set_ylim(0, 1.05)                  # Ratio (fixed)


        # Return the updated plot elements
        return self.straight_line_plot, self.deviation_plot, self.ratio_plot # CHANGED

    def close(self):
        """Close the plot window."""
        print("Attempting to close plotter window...")
        # Stop the animation first
        if hasattr(self, 'ani'):
             try:
                   self.ani.event_source.stop()
                   print("Animation stopped.")
             except AttributeError:
                   print("Could not stop animation source.")

        # Then close the figure
        if hasattr(self, 'fig') and self.fig and plt.fignum_exists(self.fig.number):
             plt.close(self.fig)
             print(f"Plotter window (fig {self.fig.number}) closed.")
        else:
             print("Plotter figure not found or already closed.")


class CameraStream:
    def __init__(self, camera_index=0):
        """ Initializes camera stream with processing and analysis features."""
        self.camera_index = camera_index
        self.cap = None
        self.frame_counter = 0
        self.tick_frequency = cv2.getTickFrequency()
        self.last_fps_update_time = time.time()
        self.accumulated_frames = 0
        self.fps = 0.0
        self.original_dimensions = (None, None)

        # --- ROI Attributes ---
        self.roi_x = None
        self.roi_y = None
        self.roi_w = None
        self.roi_h = None
        self.roi_enabled = True
        self.roi_color = (255, 255, 0) # Cyan

        # Processing flags
        self.grayscale = False
        self.flip_horizontal = False
        self.flip_vertical = False
        self.blur = False
        self.edge_detection = False
        self.display_info = True
        self.record = False
        self.writer = None

        # Line analysis flags and metrics
        self.analyze_lines = False
        self.straight_line_count = 0
        self.average_deviation_metric = 0.0 # <<<< METRIC CHANGED (Avg RMS Deviation)
        self.line_ratio = 0.0 # Ratio of Straight / Total
        self.show_line_analysis = False # Draw lines on edge view
        self.show_lines_on_original = False # Draw lines on original feed

        # Edge detection parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150

        # Line detection parameters
        self.hough_rho = 1
        self.hough_theta = np.pi/180
        self.hough_threshold = 50
        self.hough_min_line_length = 50
        self.hough_max_line_gap = 10
        self.deviation_threshold = 1.0 # Threshold for RMS deviation (pixels) to classify as 'curved' # CHANGED
        # Tune deviation threshold with ,/. keys

        # Window settings
        self.window_name = "Camera Stream (ROI Controls: IJKL/WASD, Shift+IJKL/WASD)" # Updated controls hint
        self.scale_factor = 1.0

        # Data visualization
        self.show_charts = False
        self.plotter = None
        self.last_plot_update_time = 0
        self.plot_update_interval = 0.2 # Update plot data every 0.2s

    def start(self):
        """Start the camera stream and initialize ROI."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera with index {self.camera_index}")
            if self.camera_index == 0:
                cameras = list_available_cameras()
                if cameras: print(f"Available cameras: {cameras}")
                else: print("No cameras detected.")
            return False

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.original_dimensions = (width, height)

        # Initialize ROI
        if width > 0 and height > 0:
            self.roi_w = width // 2
            self.roi_h = height // 2
            self.roi_x = (width - self.roi_w) // 2
            self.roi_y = (height - self.roi_h) // 2
            print(f"Default ROI set to: x={self.roi_x}, y={self.roi_y}, w={self.roi_w}, h={self.roi_h}")
        else:
            print("Warning: Could not get frame dimensions to set default ROI.")
            self.roi_enabled = False
            print("ROI processing disabled due to missing dimensions.")

        print(f"Camera opened: Index {self.camera_index} | Res: {width}x{height} | Reported FPS: {cam_fps:.2f}")
        print("\n--- Controls ---")
        print(" q       - Quit")
        print(" g       - Toggle Grayscale")
        print(" h       - Flip Horizontal")
        print(" v       - Flip Vertical")
        print(" b       - Toggle Gaussian Blur (RECOMMENDED for line analysis)")
        print(" e       - Toggle Edge Detection View (Canny)")
        print(" l       - Toggle Line Analysis (Calculates straight/deviation in ROI)") # Updated description
        print(" a       - Toggle Line Analysis Visualization (on Edge View)")
        print(" o       - Toggle Line Analysis Visualization (on Processed View)")
        print(" c       - Toggle Real-time Charts (enables Line Analysis if off)")
        print(" i       - Toggle Info Display")
        print(" r       - Toggle Recording (.avi)")
        print(" s       - Save Snapshot (.jpg)")
        print(" +/-     - Increase/Decrease Window Size")
        print(" [/]     - Decrease/Increase Canny Low Threshold")
        print(" {/}     - Decrease/Increase Canny High Threshold")
        print(" </>/.   - Decrease/Increase RMS Deviation Threshold") # CHANGED description
        print(" n/m     - Decrease/Increase Hough Min Line Length")
        print(" j/k     - Decrease/Increase Hough Max Line Gap")
        print(" R       - Toggle ROI Analysis ON/OFF")
        print(" WASD    - Move ROI (1px)")
        print(" IJKL    - Move ROI (5px)")
        print(" Shift+W/S/A/D - Resize ROI (2px)")
        print(" Shift+I/K/J/L - Resize ROI (10px)")
        print("----------------")

        self.last_fps_update_time = time.time()
        self.accumulated_frames = 0
        return True

    def analyze_edge_lines(self, edges_roi):
        """
        Analyze edges *within the ROI* using HoughLinesP and classify lines,
        calculating average RMS deviation for 'curved' (high deviation) lines.
        Returns:
            tuple: (straight_count, deviated_line_count, avg_deviation, lines_data)
                   where lines_data is a list of tuples:
                   [(x1_roi, y1_roi, x2_roi, y2_roi, classification, deviation_metric), ...]
                   Coordinates are relative to the ROI. Deviation metric added to data.
        """
        lines = cv2.HoughLinesP(
            edges_roi, # Operate on the cropped edges
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        straight_lines = 0
        deviated_lines_found_count = 0 # Count lines classified as having high deviation
        sum_of_deviations = 0.0      # Accumulate RMS deviation for averaging
        lines_data = [] # Store line info: (x1, y1, x2, y2, type, deviation_metric) RELATIVE TO ROI

        if lines is None:
            # Update counts and metric directly if no lines found
            self.straight_line_count = 0
            self.average_deviation_metric = 0.0
            self.line_ratio = 0.0
            return 0, 0, 0.0, [] # Return zero counts, zero deviation, empty list

        for line in lines:
            x1, y1, x2, y2 = line[0] # These are relative to edges_roi
            line_length_sq = (x2 - x1)**2 + (y2 - y1)**2

            classification = 'straight' # Default
            deviation_metric = 0.0 # Initialize deviation metric for this line

            # Skip deviation check for very short segments (treat as straight)
            min_length_for_analysis = 5 # Pixels
            if line_length_sq < min_length_for_analysis**2:
                straight_lines += 1
            else:
                # Pass edges_roi to get_line_points
                points = self.get_line_points(edges_roi, x1, y1, x2, y2)

                if len(points) < 3: # Need > 2 points for deviation check
                    straight_lines += 1
                else:
                    # Calculate RMS deviation metric
                    deviation_metric = self.calculate_rms_deviation(points) # CHANGED function call

                    # Classify based on the deviation threshold
                    if deviation_metric < self.deviation_threshold: # CHANGED threshold variable
                        straight_lines += 1
                    else:
                        # This line has high deviation (classified as 'curved'/'deviated')
                        classification = 'deviated' # CHANGED classification name for clarity
                        # Add deviation metric and count for averaging
                        sum_of_deviations += deviation_metric
                        deviated_lines_found_count += 1

            # Store line data with coordinates relative to ROI, classification, and deviation metric
            lines_data.append((x1, y1, x2, y2, classification, deviation_metric)) # ADD deviation_metric

        # --- Update counts, average deviation metric, and ratio based on ROI analysis ---
        self.straight_line_count = straight_lines
        # Calculate average deviation metric
        self.average_deviation_metric = sum_of_deviations / deviated_lines_found_count if deviated_lines_found_count > 0 else 0.0

        # Update total lines count based on straight + actual deviated found
        total_lines = straight_lines + deviated_lines_found_count
        self.line_ratio = straight_lines / total_lines if total_lines > 0 else 0.0

        # --- Update plotter data if enabled ---
        current_time = time.time()
        if self.plotter and self.show_charts and (current_time - self.last_plot_update_time > self.plot_update_interval):
             # Pass average deviation metric
             self.plotter.add_data_point(
                 self.straight_line_count, self.average_deviation_metric, self.line_ratio
             )
             self.last_plot_update_time = current_time

        # Return counts, average deviation, and the detailed line data (ROI coordinates)
        return straight_lines, deviated_lines_found_count, self.average_deviation_metric, lines_data


    def get_line_points(self, edges, x1, y1, x2, y2):
        """
        Extracts edge points along a line segment using Bresenham-like logic.
        Operates on the provided edge map (which should be the ROI edge map).
        Coordinates (x1,y1,x2,y2) are relative to the 'edges' map.
        """
        points = []
        h, w = edges.shape[:2] # Get dimensions from the input edge map
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        x_curr, y_curr = x1, y1

        # Bresenham algorithm adapted to collect points
        while True:
            # Check bounds *inside* the loop
            if 0 <= x_curr < w and 0 <= y_curr < h:
                if edges[y_curr, x_curr] > 0: # Check if the pixel is an edge point
                    points.append((x_curr, y_curr))
            else:
                 # Optional: Stop if we step outside bounds?
                 # Or continue Bresenham logic but don't add points?
                 # For simplicity, let's just check inside. If start is OOB, list is empty.
                 pass # Continue logic below to reach endpoint check

            if x_curr == x2 and y_curr == y2: break # Reached end point

            # Bresenham algorithm steps
            e2 = 2 * err
            if e2 >= dy:
                if x_curr == x2: break # Prevent infinite loop on vertical lines if endpoint check failed
                err += dy
                x_curr += sx
            if e2 <= dx:
                if y_curr == y2: break # Prevent infinite loop on horizontal lines if endpoint check failed
                err += dx
                y_curr += sy
            # Safety break if something goes wrong (e.g., coords explode)
            if abs(x_curr - x1) > w*2 or abs(y_curr - y1) > h*2: # Heuristic break
                # print("Warning: Bresenham loop runaway detected.")
                break

        # Ensure endpoints are included if they were edge points (Bresenham might miss last step)
        # Check start point (if not already added)
        if 0 <= x1 < w and 0 <= y1 < h and edges[y1, x1] > 0 and (x1, y1) not in points:
             points.insert(0, (x1, y1)) # Add at beginning if missed
        # Check end point (if not already added)
        if 0 <= x2 < w and 0 <= y2 < h and edges[y2, x2] > 0 and (x2, y2) not in points:
             points.append((x2, y2)) # Add at end if missed


        return points


    def calculate_rms_deviation(self, points):
        """
        Calculates the Root Mean Square (RMS) deviation of the points
        from the straight line segment connecting the first and last point.
        Returns the RMS deviation in pixels.
        """
        if len(points) < 3:
            return 0.0

        p_start = np.array(points[0])
        p_end = np.array(points[-1])
        line_vec = p_end - p_start
        segment_length_sq = np.dot(line_vec, line_vec) # Use squared length to avoid sqrt until end

        if segment_length_sq < 1e-6: # Avoid division by zero for coincident start/end points
            return 0.0

        sum_sq_distances = 0.0
        num_points = len(points)

        # Iterate through all points (including start and end, though their distance should be ~0)
        for i in range(num_points):
            p_i = np.array(points[i])
            point_vec = p_i - p_start

            # Project point_vec onto line_vec
            # Formula for distance from point p_i to line segment p_start -> p_end:
            # t = dot(p_i - p_start, p_end - p_start) / |p_end - p_start|^2
            # If 0 <= t <= 1, projection is onto the segment.
            # Closest point on line = p_start + t * (p_end - p_start)
            # Distance = |p_i - closest_point_on_line|

            # Optimized calculation using cross product (2D):
            # Perpendicular distance = |cross(p_end - p_start, p_start - p_i)| / |p_end - p_start|
            # cross(A, B) for 2D vectors (ax, ay) and (bx, by) is ax*by - ay*bx
            numerator = np.abs(line_vec[0] * (-point_vec[1]) - line_vec[1] * (-point_vec[0])) # |line_vec x (-point_vec)| = |line_vec x (p_start - p_i)|
            denominator_sq = segment_length_sq

            # distance^2 = numerator^2 / denominator^2
            # We already have denominator^2 as segment_length_sq
            distance_sq = (numerator**2) / denominator_sq
            sum_sq_distances += distance_sq

        # Calculate RMS deviation
        mean_sq_distance = sum_sq_distances
        rms_deviation = np.sqrt(mean_sq_distance)

        return rms_deviation


    def _ensure_roi_bounds(self):
        """Adjust ROI coordinates to stay within frame boundaries."""
        frame_w, frame_h = self.original_dimensions
        if frame_w is None or frame_h is None or frame_w <= 0 or frame_h <= 0:
            self.roi_enabled = False # Cannot validate bounds
            return

        if self.roi_x is None: # Initialize if somehow still None
             self.roi_w = frame_w // 2
             self.roi_h = frame_h // 2
             self.roi_x = (frame_w - self.roi_w) // 2
             self.roi_y = (frame_h - self.roi_h) // 2

        # Ensure minimum size
        min_roi_w = 10
        min_roi_h = 10
        self.roi_w = max(min_roi_w, self.roi_w)
        self.roi_h = max(min_roi_h, self.roi_h)

        # Constrain position and size
        self.roi_x = max(0, self.roi_x)
        self.roi_y = max(0, self.roi_y)
        # Adjust size first based on potential position overflow
        self.roi_w = min(self.roi_w, frame_w - self.roi_x)
        self.roi_h = min(self.roi_h, frame_h - self.roi_y)
        # Re-check position based on clamped size (shouldn't be needed if position clamped first)
        # self.roi_x = min(self.roi_x, frame_w - self.roi_w)
        # self.roi_y = min(self.roi_y, frame_h - self.roi_h)

        # Ensure min size constraint didn't push it out again (if size clamping reduced below min)
        self.roi_w = max(min_roi_w, self.roi_w)
        self.roi_h = max(min_roi_h, self.roi_h)
        # Final position check based on potentially re-enlarged min size
        self.roi_x = min(self.roi_x, frame_w - self.roi_w)
        self.roi_y = min(self.roi_y, frame_h - self.roi_h)


    def process_frame(self, frame):
        """Apply selected processing to the frame, using ROI for analysis."""
        if self.roi_enabled:
            self._ensure_roi_bounds()

        processed_frame = frame.copy()
        if self.flip_horizontal: processed_frame = cv2.flip(processed_frame, 1)
        if self.flip_vertical: processed_frame = cv2.flip(processed_frame, 0)

        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        analysis_source = gray
        if self.blur:
            # Use a smaller kernel for potentially better detail preservation with RMS deviation?
            # Or keep larger for noise reduction? Let's keep 15x15 for now.
            analysis_source = cv2.GaussianBlur(gray, (15, 15), 0)

        # --- ROI Handling & Analysis ---
        lines_data_roi = []
        edges_roi = None
        # Store analysis results locally for clarity if needed, though self attributes are updated too
        analyzed_straight_count = 0
        analyzed_deviated_count = 0
        analyzed_avg_deviation = 0.0

        if self.analyze_lines and self.roi_enabled and self.roi_w > 0 and self.roi_h > 0:
            roi_to_analyze = analysis_source[self.roi_y : self.roi_y + self.roi_h,
                                             self.roi_x : self.roi_x + self.roi_w]
            edges_roi = cv2.Canny(roi_to_analyze, self.canny_threshold1, self.canny_threshold2)

            # Analyze lines within edges_roi - updates self attributes and returns values
            analyzed_straight_count, analyzed_deviated_count, analyzed_avg_deviation, lines_data_roi = self.analyze_edge_lines(edges_roi)

        elif self.analyze_lines: # Analysis enabled but ROI is not
            # Reset counts and deviation if analysis is expected but not performed
            self.straight_line_count = 0
            self.average_deviation_metric = 0.0
            self.line_ratio = 0.0
            # Optionally analyze full frame here as fallback (not implemented)

        # --- Determine Base Display Frame ---
        display_frame = processed_frame
        if self.grayscale:
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # Use original gray if displaying grayscale

        if self.edge_detection:
            # Create a black canvas and place the ROI edges onto it
            full_edges_display = np.zeros_like(processed_frame)
            if edges_roi is not None: # Edges were calculated for ROI
                 edges_roi_bgr = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)
                 # Ensure dimensions match before assignment (should be guaranteed by roi_w/h)
                 if edges_roi_bgr.shape[0] == self.roi_h and edges_roi_bgr.shape[1] == self.roi_w:
                      full_edges_display[self.roi_y : self.roi_y + self.roi_h,
                                         self.roi_x : self.roi_x + self.roi_w] = edges_roi_bgr
                 else:
                      print(f"Warning: Mismatch edge ROI shape {edges_roi_bgr.shape[:2]} vs ROI dims {self.roi_h}x{self.roi_w}")
            elif self.analyze_lines and not self.roi_enabled: # Analysis on but ROI off, show full edges
                 edges_full = cv2.Canny(analysis_source, self.canny_threshold1, self.canny_threshold2)
                 full_edges_display = cv2.cvtColor(edges_full, cv2.COLOR_GRAY2BGR)
            # If edge view is on but analysis is off, show full edges
            elif not self.analyze_lines:
                 edges_full = cv2.Canny(analysis_source, self.canny_threshold1, self.canny_threshold2)
                 full_edges_display = cv2.cvtColor(edges_full, cv2.COLOR_GRAY2BGR)

            display_frame = full_edges_display # Display the canvas with edges


        # --- Draw Lines (if analysis occurred and visualization enabled) ---
        if lines_data_roi: # Check if analysis produced line data
              draw_on_edge_view = self.edge_detection and self.show_line_analysis
              draw_on_processed_view = not self.edge_detection and self.show_lines_on_original

              if draw_on_edge_view or draw_on_processed_view:
                   # Target the correct frame for drawing
                   target_frame_for_lines = display_frame # Draw on whatever is currently being displayed

                   # Unpack the extended lines_data tuple (now includes deviation metric)
                   for x1_roi, y1_roi, x2_roi, y2_roi, classification, deviation_metric in lines_data_roi:
                        x1_global = x1_roi + self.roi_x
                        y1_global = y1_roi + self.roi_y
                        x2_global = x2_roi + self.roi_x
                        y2_global = y2_roi + self.roi_y

                        # Color based on classification
                        color = (0, 255, 0) if classification == 'straight' else (0, 0, 255) # Green for straight, Red for deviated
                        # Optional: Vary color/thickness based on deviation_metric value for deviated lines
                        # thickness = 1 + int(deviation_metric) # Example scaling
                        cv2.line(target_frame_for_lines, (x1_global, y1_global), (x2_global, y2_global), color, 1) # Use thinner lines?

        # --- Draw ROI Rectangle ---
        # Draw ROI on the *final* display frame, after potential edge view setup
        if self.roi_enabled:
            cv2.rectangle(display_frame,
                          (self.roi_x, self.roi_y),
                          (self.roi_x + self.roi_w, self.roi_y + self.roi_h),
                          self.roi_color, 1)

        # --- Calculate FPS ---
        current_time = time.time()
        self.accumulated_frames += 1
        time_diff = current_time - self.last_fps_update_time
        if time_diff >= 1.0:
            self.fps = self.accumulated_frames / time_diff
            self.last_fps_update_time = current_time
            self.accumulated_frames = 0

        # --- Add Info Overlay ---
        if self.display_info:
            # Add info overlay onto the final display_frame
            self.add_info_overlay(display_frame)

        # --- Apply Scaling ---
        if self.scale_factor != 1.0:
             height, width = display_frame.shape[:2]
             # Prevent scaling to zero dimensions
             new_width = max(10, int(width * self.scale_factor))
             new_height = max(10, int(height * self.scale_factor))
             if new_width > 0 and new_height > 0:
                  display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
             else:
                  print(f"Warning: Invalid resize dimensions ({new_width}x{new_height}), skipping resize.")

        # Return original processed frame (for recording, before overlay/scaling) and final display frame
        return processed_frame, display_frame


    def add_info_overlay(self, frame_to_draw_on):
        """Adds text information overlay to the frame."""
        height, width = frame_to_draw_on.shape[:2]
        y_offset = 25; x_offset = 10; font_scale = 0.6
        font_face = cv2.FONT_HERSHEY_SIMPLEX; font_color = (0, 255, 0) # Green
        bg_color = (0,0,0); font_thickness = 1; line_spacing = 25

        # Helper to draw text
        def draw_text(img, text, pos):
            nonlocal y_offset
            # Optional: Add background rect for better readability
            # t_size = cv2.getTextSize(text, font_face, font_scale, font_thickness)[0]
            # cv2.rectangle(img, (pos[0] - 2, pos[1] - t_size[1] - 2), (pos[0] + t_size[0] + 2, pos[1] + 2), bg_color, -1)
            cv2.putText(img, text, pos, font_face, font_scale, font_color, font_thickness, cv2.LINE_AA)
            y_offset += line_spacing

        # Basic Info
        draw_text(frame_to_draw_on, f"FPS: {self.fps:.1f}", (x_offset, y_offset))
        ow, oh = self.original_dimensions
        if ow and oh: draw_text(frame_to_draw_on, f"Res: {ow}x{oh}", (x_offset, y_offset))
        else: draw_text(frame_to_draw_on, "Res: Unknown", (x_offset, y_offset))

        # ROI Info
        if self.roi_enabled:
            roi_text = f"ROI: ({self.roi_x},{self.roi_y}) {self.roi_w}x{self.roi_h}"
            draw_text(frame_to_draw_on, roi_text, (x_offset, y_offset))
        else:
            draw_text(frame_to_draw_on, "ROI: [Disabled]", (x_offset, y_offset))

        # Effects Status
        effects = []
        if self.grayscale: effects.append("Gray")
        if self.flip_horizontal: effects.append("HFlip")
        if self.flip_vertical: effects.append("VFlip")
        if self.blur: effects.append("Blur")
        if self.edge_detection: effects.append("Edges")
        if self.analyze_lines: effects.append("LineAn")
        if self.analyze_lines and self.show_line_analysis: effects.append("LineVis(E)")
        if self.analyze_lines and self.show_lines_on_original: effects.append("LineVis(O)")
        if self.show_charts: effects.append("Charts")
        if not self.roi_enabled and self.analyze_lines: effects.append("[ROI OFF]")
        if effects: draw_text(frame_to_draw_on, "FX: " + ", ".join(effects), (x_offset, y_offset))

        # Line Analysis Metrics & Parameters
        if self.analyze_lines:
            prefix = "(ROI) " if self.roi_enabled else "(Full) " # Indicate source
            draw_text(frame_to_draw_on, f"{prefix}Straight Lines: {self.straight_line_count}", (x_offset, y_offset))
            # --- DISPLAY AVG RMS DEVIATION ---
            draw_text(frame_to_draw_on, f"{prefix}Deviation: {self.average_deviation_metric:.3f} px", (x_offset, y_offset)) # CHANGED
            # --- END DISPLAY CHANGE ---
            draw_text(frame_to_draw_on, f"{prefix}Ratio S/T: {self.line_ratio:.3f}", (x_offset, y_offset))
            draw_text(frame_to_draw_on, f"Dev Thresh: {self.deviation_threshold:.2f} px", (x_offset, y_offset)) # CHANGED
            draw_text(frame_to_draw_on, f"Canny: {self.canny_threshold1}/{self.canny_threshold2}", (x_offset, y_offset))
            draw_text(frame_to_draw_on, f"Hough MinLen: {self.hough_min_line_length}", (x_offset, y_offset))
            draw_text(frame_to_draw_on, f"Hough MaxGap: {self.hough_max_line_gap}", (x_offset, y_offset))

        # Recording Indicator (Top Right)
        if self.record:
            rec_text = "REC"
            text_size, _ = cv2.getTextSize(rec_text, font_face, 0.5, 1)
            text_w, text_h = text_size
            rec_pos_x = width - text_w - 15
            rec_pos_y = 25
            # Draw circle indicator slightly offset
            cv2.circle(frame_to_draw_on, (rec_pos_x - 10, rec_pos_y - text_h // 4), 5, (0, 0, 255), -1)
            cv2.putText(frame_to_draw_on, rec_text, (rec_pos_x, rec_pos_y), font_face, 0.5,(0, 0, 255), 1, cv2.LINE_AA)


    def run(self):
        """Run the main camera stream loop."""
        if not self.start(): return

        cv2.namedWindow(self.window_name)

        last_plt_pause_time = time.time()
        plt_pause_interval = 0.05 # How often to yield to matplotlib GUI

        try:
            while True:
                # --- Matplotlib GUI Update ---
                if self.show_charts and self.plotter and hasattr(self.plotter, 'fig') and self.plotter.fig.canvas:
                     current_time = time.time()
                     if current_time - last_plt_pause_time > plt_pause_interval:
                          plt.pause(0.001) # Allow GUI event processing
                          last_plt_pause_time = current_time

                     # Check if plotter window was closed by user
                     # Check existence before accessing number attribute
                     if not hasattr(self.plotter,'fig') or not self.plotter.fig or not plt.fignum_exists(self.plotter.fig.number):
                          print("Plotter window closed by user or invalid. Disabling charts.")
                          self.show_charts = False
                          if self.plotter: self.plotter.close() # Ensure resources are released
                          self.plotter = None


                # --- Read Camera Frame ---
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Warning: Failed to capture frame.")
                    # Add check if camera is still open, if not, break loop
                    if not self.cap.isOpened():
                        print("Camera disconnected. Exiting.")
                        break
                    time.sleep(0.1) # Wait a bit before retrying
                    continue

                # --- Process Frame ---
                original_processed_frame, display_frame = self.process_frame(frame)

                # --- Handle Recording ---
                if self.record:
                    if self.writer is None: # Start recording
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_filename = f"camera_rec_{timestamp}.avi"
                        # Record the original processed frame dimensions
                        rec_h, rec_w = original_processed_frame.shape[:2]
                        if rec_w > 0 and rec_h > 0:
                            try:
                                # Use a sensible FPS for recording, capped at 30
                                rec_fps = max(5.0, min(self.fps, 30.0) if self.fps > 0 else 20.0)
                                self.writer = cv2.VideoWriter(output_filename, fourcc, rec_fps, (rec_w, rec_h))
                                if not self.writer.isOpened(): raise IOError("VideoWriter failed to open")
                                print(f"Recording started: {output_filename} at ~{rec_fps:.1f} FPS ({rec_w}x{rec_h})")
                            except Exception as e:
                                print(f"Error initializing VideoWriter: {e}")
                                self.record = False # Turn off recording flag
                                if self.writer: self.writer.release(); self.writer = None # Clean up writer if partially created
                        else:
                            print("Error: Cannot record with invalid frame dimensions.")
                            self.record = False

                    # If writer exists and is open, write the frame
                    if self.writer is not None and self.writer.isOpened():
                         try:
                             # Write the frame *before* overlays/scaling are added
                             self.writer.write(original_processed_frame)
                         except Exception as e:
                             print(f"Error writing frame: {e}")
                             # Consider stopping recording on write error?
                             # self.record = False
                elif not self.record and self.writer is not None: # Stop recording if flag is off but writer exists
                    print("Stopping recording...")
                    self.writer.release()
                    self.writer = None
                    print("Recording stopped.")

                # --- Display Frame ---
                # Ensure display_frame is not empty before showing
                if display_frame is not None and display_frame.size > 0:
                    cv2.imshow(self.window_name, display_frame)
                else:
                    print("Warning: display_frame is empty, skipping imshow.")


                # --- Handle Keyboard Input ---
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key != 0xFF: # Key pressed
                    # Pass the frame *before* overlay/scaling for snapshot
                    self.handle_key_press(key, original_processed_frame)

        finally:
            print("Cleaning up...")
            if self.cap is not None:
                self.cap.release()
                print("Camera released.")
            if self.writer is not None and self.writer.isOpened():
                print("Releasing active video writer...")
                self.writer.release()
                print("Video writer released.")

            if self.plotter is not None:
                self.plotter.close()
                self.plotter = None

            cv2.destroyAllWindows()
            print("OpenCV windows destroyed.")
            # Explicitly call plt.close('all') in case run finishes while plot open
            plt.close('all')
            print("Matplotlib windows closed.")


    def handle_key_press(self, key, frame_for_snapshot):
        """Handles key presses for changing settings and ROI."""
        adjust_roi_pos = False
        adjust_roi_size = False
        shift_pressed = False # Simple proxy - check capital letters for Shift+key

        # Check for Shift (using capital letters for resize keys)
        if ord('A') <= key <= ord('Z'):
             # Check if it's one of our Shift+key combos
             if key in [ord('W'), ord('S'), ord('A'), ord('D'), ord('I'), ord('K'), ord('J'), ord('L')]:
                   shift_pressed = True


        # --- Basic Toggles ---
        if key == ord('g'): self.grayscale = not self.grayscale; print(f"Grayscale: {self.grayscale}")
        elif key == ord('h'): self.flip_horizontal = not self.flip_horizontal; print(f"H-Flip: {self.flip_horizontal}")
        elif key == ord('v'): self.flip_vertical = not self.flip_vertical; print(f"V-Flip: {self.flip_vertical}")
        elif key == ord('b'): self.blur = not self.blur; print(f"Blur: {self.blur}")
        elif key == ord('e'): self.edge_detection = not self.edge_detection; print(f"Edge View: {self.edge_detection}")
        elif key == ord('i') and not shift_pressed: self.display_info = not self.display_info; print(f"Info: {self.display_info}") # Avoid clash with Shift+I
        elif key == ord('l') and not shift_pressed: # Avoid clash with Shift+L
            self.analyze_lines = not self.analyze_lines; print(f"Line Analysis: {self.analyze_lines}")
            if not self.analyze_lines:
                self.show_line_analysis = False; print("Line Vis (Edge): Off")
                self.show_lines_on_original = False; print("Line Vis (Orig): Off")
                # Reset counts AND deviation metric when turning off analysis
                self.straight_line_count = 0
                self.average_deviation_metric = 0.0 # <<<< RESET
                self.line_ratio = 0.0
        elif key == ord('a') and not shift_pressed: # Avoid clash with Shift+A
            if self.analyze_lines:
                self.show_line_analysis = not self.show_line_analysis
                print(f"Line Vis (Edge): {self.show_line_analysis}")
                if self.show_line_analysis: self.show_lines_on_original = False # Mutually exclusive vis
            else: print("Enable Line Analysis ('l') first.")
        elif key == ord('o'): # Toggle line vis on original view
             if self.analyze_lines:
                  self.show_lines_on_original = not self.show_lines_on_original
                  print(f"Line Vis (Orig): {self.show_lines_on_original}")
                  if self.show_lines_on_original: self.show_line_analysis = False # Mutually exclusive vis
             else: print("Enable Line Analysis ('l') first.")
        elif key == ord('c'): # Toggle charts
            self.show_charts = not self.show_charts
            if self.show_charts:
                # Check if plotter exists and its figure window is still open
                plotter_open = False
                if self.plotter and hasattr(self.plotter, 'fig') and self.plotter.fig and plt.fignum_exists(self.plotter.fig.number):
                     plotter_open = True

                if not plotter_open: # Recreate if closed or never created
                     print("Initializing plotter...")
                     if not plt.isinteractive(): plt.ion() # Ensure interactive mode
                     self.plotter = DataPlotter()
                     self.last_plot_update_time = time.time() # Reset timer
                     try: # Force redraw/update
                          # self.plotter.fig.canvas.draw_idle() # Might not be needed with FuncAnimation
                          plt.pause(0.01) # Give it a moment to appear
                     except Exception as e: print(f"Plotter init display error: {e}")

                if not self.analyze_lines:
                     self.analyze_lines = True; print("Line analysis auto ON for charts.")
                # Add a point immediately if plotter was just created or reopened
                if self.plotter: # Check again if it was successfully created
                     self.plotter.add_data_point(self.straight_line_count, self.average_deviation_metric, self.line_ratio) # <<<< Pass deviation metric
                print("Charts: On")
            else: # Turning charts off
                print("Charts: Off")
                if self.plotter:
                     self.plotter.close() # Properly close the plotter window
                     self.plotter = None

        elif key == ord('r'): self.record = not self.record # Toggle recording flag (actual start/stop handled in loop)
        elif key == ord('s') and not shift_pressed: # Avoid clash with Shift+S
             timestamp = time.strftime("%Y%m%d_%H%M%S")
             filename = f"camera_snapshot_{timestamp}.jpg"
             # Save the frame passed in (before overlay/scaling)
             cv2.imwrite(filename, frame_for_snapshot)
             print(f"Snapshot saved: {filename}")
        elif key == ord('R'): # Toggle ROI analysis enable/disable
             self.roi_enabled = not self.roi_enabled
             print(f"ROI Analysis Enabled: {self.roi_enabled}")
             if not self.roi_enabled:
                 # Reset ROI-specific counts and deviation metric when disabling ROI
                 # (Analysis might switch to full frame if implemented, or just show zeros)
                 self.straight_line_count = 0
                 self.average_deviation_metric = 0.0 # <<<< RESET
                 self.line_ratio = 0.0

        # --- Window Scaling ---
        elif key == ord('+') or key == ord('='): self.scale_factor = min(3.0, self.scale_factor + 0.1); print(f"Scale: {self.scale_factor:.1f}x")
        elif key == ord('-') or key == ord('_'): self.scale_factor = max(0.1, self.scale_factor - 0.1); print(f"Scale: {self.scale_factor:.1f}x") # Min scale 0.1

        # --- Canny Tuning ---
        elif key == ord('['): self.canny_threshold1 = max(0, self.canny_threshold1 - 10); self.canny_threshold2 = max(self.canny_threshold1 + 10, self.canny_threshold2); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")
        elif key == ord(']'): self.canny_threshold1 = min(self.canny_threshold2 - 10, self.canny_threshold1 + 10); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}") # Ensure T1 < T2
        elif key == ord('{'): self.canny_threshold2 = max(self.canny_threshold1 + 10, self.canny_threshold2 - 10); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}") # Ensure T2 > T1
        elif key == ord('}'): self.canny_threshold2 = min(255, self.canny_threshold2 + 10); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")

        # --- RMS Deviation Threshold Tuning --- # CHANGED section
        elif key == ord(',') or key == ord('<'): self.deviation_threshold = max(0.1, self.deviation_threshold - 0.1); print(f"Deviation Thresh: {self.deviation_threshold:.2f} px")
        elif key == ord('.') or key == ord('>'): self.deviation_threshold += 0.1; print(f"Deviation Thresh: {self.deviation_threshold:.2f} px")

        # --- Hough Tuning ---
        elif key == ord('n'): self.hough_min_line_length = max(1, self.hough_min_line_length - 5); print(f"Hough MinLen: {self.hough_min_line_length}")
        elif key == ord('m'): self.hough_min_line_length += 5; print(f"Hough MinLen: {self.hough_min_line_length}")
        elif key == ord('j') and not shift_pressed: self.hough_max_line_gap = max(0, self.hough_max_line_gap - 2); print(f"Hough MaxGap: {self.hough_max_line_gap}") # Avoid clash Shift+J
        elif key == ord('k') and not shift_pressed: self.hough_max_line_gap += 2; print(f"Hough MaxGap: {self.hough_max_line_gap}") # Avoid clash Shift+K

        # --- ROI Controls ---
        if self.roi_x is not None and self.roi_enabled: # Only adjust if ROI exists and is enabled
              step_fine = 1
              step_coarse = 5
              resize_fine = 2
              resize_coarse = 10

              # Fine Movement (WASD - lowercase)
              if not shift_pressed:
                  if key == ord('w'): self.roi_y -= step_fine; adjust_roi_pos = True
                  elif key == ord('s'): self.roi_y += step_fine; adjust_roi_pos = True
                  elif key == ord('a'): self.roi_x -= step_fine; adjust_roi_pos = True
                  elif key == ord('d'): self.roi_x += step_fine; adjust_roi_pos = True
              # Coarse Movement (IJKL - lowercase)
                  elif key == ord('i'): self.roi_y -= step_coarse; adjust_roi_pos = True
                  elif key == ord('k'): self.roi_y += step_coarse; adjust_roi_pos = True
                  elif key == ord('j'): self.roi_x -= step_coarse; adjust_roi_pos = True
                  elif key == ord('l'): self.roi_x += step_coarse; adjust_roi_pos = True

              # Fine Resize (Shift + WASD - check capital)
              if shift_pressed:
                  if key == ord('W'): self.roi_h -= resize_fine; adjust_roi_size = True
                  elif key == ord('S'): self.roi_h += resize_fine; adjust_roi_size = True
                  elif key == ord('A'): self.roi_w -= resize_fine; adjust_roi_size = True
                  elif key == ord('D'): self.roi_w += resize_fine; adjust_roi_size = True
              # Coarse Resize (Shift + IJKL - check capital)
                  elif key == ord('I'): self.roi_h -= resize_coarse; adjust_roi_size = True
                  elif key == ord('K'): self.roi_h += resize_coarse; adjust_roi_size = True
                  elif key == ord('J'): self.roi_w -= resize_coarse; adjust_roi_size = True
                  elif key == ord('L'): self.roi_w += resize_coarse; adjust_roi_size = True

              # Ensure bounds after adjustment
              if adjust_roi_pos or adjust_roi_size:
                   # print(f"ROI Adjust: Pos={adjust_roi_pos}, Size={adjust_roi_size}") # Debug
                   self._ensure_roi_bounds()
                   print(f"New ROI: x={self.roi_x}, y={self.roi_y}, w={self.roi_w}, h={self.roi_h}")
                   # Optional: Reset analysis when ROI changes? Might be too jumpy.
                   # self.straight_line_count = 0
                   # self.average_deviation_metric = 0.0
                   # self.line_ratio = 0.0


# --- Helper Function ---
def list_available_cameras(max_check=10):
    """List available camera indices that can be opened and read from."""
    available_cameras = []
    print(f"Checking for cameras up to index {max_check-1}...")
    for i in range(max_check):
        cap_test = cv2.VideoCapture(i)
        is_opened = cap_test.isOpened()
        can_read = False
        if is_opened:
            # Try reading a frame as a more robust check
            ret, _ = cap_test.read()
            if ret:
                can_read = True
                available_cameras.append(i)
            # else:
                # print(f"Index {i} opened but failed to read frame.") # Optional debug
        if cap_test is not None: # Ensure release happens
            cap_test.release()
        # print(f"Index {i}: Opened={is_opened}, Readable={can_read}") # Optional debug
    return available_cameras

# --- Main Execution ---
if __name__ == "__main__":
    camera_index = 0
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            cameras = list_available_cameras()
            if not cameras: print("No cameras detected or none could be read from.")
            else: print(f"Available and readable camera indices: {cameras}")
            sys.exit(0)
        else:
            try: camera_index = int(sys.argv[1])
            except ValueError: print(f"Error: Invalid camera index '{sys.argv[1]}'. Use '--list' to see available cameras."); sys.exit(1)

    print(f"Attempting to use camera index: {camera_index}")
    stream = CameraStream(camera_index=camera_index)
    stream.run()
    print("Program finished.")