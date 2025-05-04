# Full code with modifications aimed at improving curved line detection robustness
# AND adding ROI functionality
import cv2
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from threading import Lock # Using Lock, assuming Thread is not strictly needed for now
import matplotlib

# --- Try using Qt5Agg backend first, often more robust for embedding ---
# Ensure PyQt5 is installed: pip install PyQt5
try:
    matplotlib.use('Qt5Agg')
    print("Using Matplotlib backend: Qt5Agg")
except ImportError:
    print("Warning: Qt5Agg backend not found or PyQt5 not installed (pip install PyQt5).")
    try:
        # Fallback to TkAgg if Qt5Agg fails
        matplotlib.use('TkAgg')
        print("Using Matplotlib backend: TkAgg")
    except ImportError:
        print("Warning: TkAgg backend not found either. Using default Matplotlib backend.")

# Set parameters after setting the backend
plt.rcParams['figure.figsize'] = [10, 8]  # Set default figure size
plt.rcParams['figure.autolayout'] = True  # Better layout
plt.rcParams['lines.linewidth'] = 2  # Thicker lines for better visibility

class DataPlotter:
    def __init__(self, max_points=100):
        """
        Initialize a real-time data plotter using matplotlib.
        Parameters: max_points (int): Max history length.
        """
        self.max_points = max_points
        self.lock = Lock()
        self.timestamps = deque(maxlen=max_points)
        self.straight_lines = deque(maxlen=max_points)
        self.curved_lines = deque(maxlen=max_points)
        self.line_ratios = deque(maxlen=max_points)

        # Initial dummy data point
        self.timestamps.append(0)
        self.straight_lines.append(0)
        self.curved_lines.append(0)
        self.line_ratios.append(0)

        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        self.fig.suptitle('Camera Stream Analysis (ROI)', fontsize=16) # Added (ROI)

        self.straight_line_plot, = self.axes[0].plot([], [], 'g-', label='Straight Lines (ROI)')
        self.curved_line_plot, = self.axes[0].plot([], [], 'r-', label='Curved Lines (ROI)')
        self.ratio_plot, = self.axes[1].plot([], [], 'b-', label='Straight/Total Ratio (ROI)')
        self.total_plot, = self.axes[2].plot([], [], 'w-', label='Total Lines (ROI)')

        self.axes[0].set_title('Line Counts (ROI)')
        self.axes[0].set_ylabel('Count')
        self.axes[0].legend(loc='upper left')
        self.axes[0].grid(True, linestyle='--', alpha=0.6)

        self.axes[1].set_title('Line Ratio (Straight / Total) (ROI)')
        self.axes[1].set_ylabel('Ratio')
        self.axes[1].set_ylim(0, 1.05)
        self.axes[1].legend(loc='upper left')
        self.axes[1].grid(True, linestyle='--', alpha=0.6)

        self.axes[2].set_title('Total Lines (Straight + Curved) (ROI)')
        self.axes[2].set_xlabel('Time (s)')
        self.axes[2].set_ylabel('Count')
        self.axes[2].legend(loc='upper left')
        self.axes[2].grid(True, linestyle='--', alpha=0.6)

        for ax in self.axes:
            ax.set_xlim(0, 10)
        self.axes[0].set_ylim(0, 10)
        self.axes[2].set_ylim(0, 10)

        self.ani = FuncAnimation(
            self.fig, self.update_plot, interval=100,
            blit=False, cache_frame_data=False
        )

        # Ensure interactive mode is on AFTER creating the animation
        if not plt.isinteractive(): plt.ion()
        plt.show(block=False)
        # Crucial: Ensure the event loop integrates. Sometimes needed, esp. with TkAgg.
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError: # Some backends might not support this
            pass
        self.start_time = time.time()

    def add_data_point(self, straight_lines, curved_lines, line_ratio):
        """Add a new data point to the plot queues."""
        with self.lock:
            current_time = time.time() - self.start_time
            # Add data point only if time has advanced to avoid duplicate timestamps
            if not self.timestamps or current_time > self.timestamps[-1] + 1e-6: # Add small epsilon
                self.timestamps.append(current_time)
                self.straight_lines.append(straight_lines)
                self.curved_lines.append(curved_lines)
                self.line_ratios.append(line_ratio)
            elif self.timestamps: # Update last point if timestamp hasn't changed significantly
                self.straight_lines[-1] = straight_lines
                self.curved_lines[-1] = curved_lines
                self.line_ratios[-1] = line_ratio


    def update_plot(self, frame):
        """Update the plot with new data. Called by FuncAnimation."""
        with self.lock:
            # Handle the initial dummy point correctly
            if len(self.timestamps) <= 1 and self.timestamps[0] == 0:
                # Return existing artists if no real data yet
                return self.straight_line_plot, self.curved_line_plot, self.ratio_plot, self.total_plot

            # Convert deques to lists for plotting
            times_list = list(self.timestamps)
            straight_list = list(self.straight_lines)
            curved_list = list(self.curved_lines)
            ratio_list = list(self.line_ratios)

            # Skip the initial dummy point (index 0) if more data exists
            if len(times_list) > 1:
                times_plot = times_list[1:]
                straight_plot = straight_list[1:]
                curved_plot = curved_list[1:]
                ratio_plot_data = ratio_list[1:]
            else: # Only the dummy point exists, nothing to plot yet
                 return self.straight_line_plot, self.curved_line_plot, self.ratio_plot, self.total_plot

        # Check if there's actual data to plot after slicing
        if not times_plot:
            return self.straight_line_plot, self.curved_line_plot, self.ratio_plot, self.total_plot

        # Set plot data
        self.straight_line_plot.set_data(times_plot, straight_plot) # Re-enabled straight line plot
        self.curved_line_plot.set_data(times_plot, curved_plot)
        self.ratio_plot.set_data(times_plot, ratio_plot_data)
        total_lines = [s + c for s, c in zip(straight_plot, curved_plot)]
        self.total_plot.set_data(times_plot, total_lines)

        # Dynamic X-axis scaling (sliding window)
        history_duration = 20.0 # Show last 20 seconds
        current_max_time = times_plot[-1] if times_plot else 0
        x_min = max(0, current_max_time - history_duration)
        # Ensure x_max is always ahead of x_min, provide buffer
        x_max = max(x_min + 10.0, current_max_time + 2.0)

        # Dynamic Y-axis scaling based on visible data
        visible_indices = [i for i, t in enumerate(times_plot) if t >= x_min]
        if not visible_indices:
            y_max_lines = 10 # Default if no data in view
            y_max_total = 10
        else:
            first_visible_index = visible_indices[0]
            # Slice data lists to get only the visible points for max calculation
            visible_straight = straight_plot[first_visible_index:]
            visible_curved = curved_plot[first_visible_index:]
            visible_total = total_lines[first_visible_index:]
            # Use default=0 in max() to handle cases where slices might become empty
            max_s = max(visible_straight, default=0)
            max_c = max(visible_curved, default=0)
            max_t = max(visible_total, default=0)
            # Ensure axes don't shrink below a minimum value (e.g., 10)
            y_max_lines = max(max_s, max_c, 10)
            y_max_total = max(max_t, 10)


        # Update axes limits
        for ax in self.axes:
             ax.set_xlim(x_min, x_max)
        # Add padding to y-limits (e.g., 20%)
        self.axes[0].set_ylim(0, y_max_lines * 1.2)
        self.axes[1].set_ylim(0, 1.05) # Ratio plot keeps fixed limits
        self.axes[2].set_ylim(0, y_max_total * 1.2)


        # Return the updated plot elements
        return self.straight_line_plot, self.curved_line_plot, self.ratio_plot, self.total_plot

    def close(self):
        """Close the plot window."""
        print("Attempting to close plotter window...")
        # Stop the animation first
        if hasattr(self, 'ani'):
             try:
                  self.ani.event_source.stop()
                  print("Animation stopped.")
             except AttributeError:
                  print("Could not stop animation source.") # Might already be stopped or backend issue

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
        self.original_dimensions = (None, None) # Placeholder

        # --- ROI Attributes ---
        self.roi_x = None
        self.roi_y = None
        self.roi_w = None
        self.roi_h = None
        self.roi_enabled = True # Control whether ROI processing is active
        self.roi_color = (255, 255, 0) # Cyan for ROI box

        # Processing flags
        self.grayscale = False
        self.flip_horizontal = False
        self.flip_vertical = False
        self.blur = False # IMPORTANT: Enable blur ('b') for better edge/line detection
        self.edge_detection = False
        self.display_info = True
        self.record = False
        self.writer = None

        # Line analysis flags
        self.analyze_lines = False
        self.straight_line_count = 0
        self.curved_line_count = 0
        self.line_ratio = 0.0
        self.show_line_analysis = False # Draw lines on edge view
        self.show_lines_on_original = False # Draw lines on original feed ('o' key)

        # Edge detection parameters (Tune with [] {} keys)
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150

        # Line detection parameters
        self.hough_rho = 1
        self.hough_theta = np.pi/180
        self.hough_threshold = 50 # Accumulator threshold
        # -- Made Tunable --
        self.hough_min_line_length = 50 # Tune with n/m keys
        self.hough_max_line_gap = 10    # Tune with j/k keys
        # -- Crucial for Classification --
        self.curvature_threshold = 0.02 # Tune with ,/. keys

        # Window settings
        self.window_name = "Camera Stream (ROI Controls: Arrows/WASD, Shift+Arrows/WASD)" # Updated name
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
                cameras = list_available_cameras() # Helper defined later
                if cameras: print(f"Available cameras: {cameras}")
                else: print("No cameras detected.")
            return False

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.original_dimensions = (width, height)

        # --- Initialize ROI (e.g., center quarter of the frame) ---
        if width > 0 and height > 0:
            self.roi_w = width // 2
            self.roi_h = height // 2
            self.roi_x = (width - self.roi_w) // 2
            self.roi_y = (height - self.roi_h) // 2
            print(f"Default ROI set to: x={self.roi_x}, y={self.roi_y}, w={self.roi_w}, h={self.roi_h}")
        else:
            print("Warning: Could not get frame dimensions to set default ROI.")
            # Disable ROI if dimensions unknown? Or set arbitrary small ones?
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
        print(" l       - Toggle Line Analysis (Calculates straight/curved in ROI)")
        print(" a       - Toggle Line Analysis Visualization (on Edge View)")
        print(" o       - Toggle Line Analysis Visualization (on Processed View)") # New
        print(" c       - Toggle Real-time Charts (enables Line Analysis if off)")
        print(" i       - Toggle Info Display")
        print(" r       - Toggle Recording (.avi)")
        print(" s       - Save Snapshot (.jpg)")
        print(" +/-     - Increase/Decrease Window Size")
        print(" [/]     - Decrease/Increase Canny Low Threshold")
        print(" {/}     - Decrease/Increase Canny High Threshold")
        print(" </>/.   - Decrease/Increase Curvature Threshold")
        print(" n/m     - Decrease/Increase Hough Min Line Length")
        print(" j/k     - Decrease/Increase Hough Max Line Gap")
        print(" R       - Toggle ROI Analysis ON/OFF") # New
        print(" Arrows  - Move ROI (5px)")
        print(" WASD    - Move ROI (1px)")
        print(" Shift+Arrows - Resize ROI (10px)")
        print(" Shift+WASD   - Resize ROI (2px)")
        print("----------------")

        self.last_fps_update_time = time.time()
        self.accumulated_frames = 0
        return True

    def analyze_edge_lines(self, edges_roi):
        """
        Analyze edges *within the ROI* using HoughLinesP and classify lines.
        Returns:
            tuple: (straight_count, curved_count, lines_data)
                   where lines_data is a list of tuples:
                   [(x1_roi, y1_roi, x2_roi, y2_roi, classification), ...]
                   Coordinates are relative to the ROI.
        """
        # No need for vis_image here, drawing happens in process_frame

        # Use tunable Hough parameters
        lines = cv2.HoughLinesP(
            edges_roi, # Operate on the cropped edges
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        straight_lines = 0
        curved_lines = 0
        lines_data = [] # Store line info: (x1, y1, x2, y2, type) RELATIVE TO ROI

        if lines is None:
            # Update counts and ratio directly if no lines found
            self.straight_line_count = 0
            self.curved_line_count = 0
            self.line_ratio = 0.0
            return 0, 0, [] # Return zero counts and empty list

        for line in lines:
            x1, y1, x2, y2 = line[0] # These are relative to edges_roi
            line_length_sq = (x2 - x1)**2 + (y2 - y1)**2

            classification = 'straight' # Default

            # Skip curvature check for very short segments (treat as straight)
            if line_length_sq < 5**2: # Consider making this threshold relative or tunable
                straight_lines += 1
                # No drawing here
            else:
                # Pass edges_roi to get_line_points
                points = self.get_line_points(edges_roi, x1, y1, x2, y2)

                if len(points) < 3: # Need > 2 points for curvature check
                    straight_lines += 1
                    # No drawing here
                else:
                    # Calculate curvature (using potentially modified method)
                    curvature = self.calculate_curvature(points)

                    if curvature < self.curvature_threshold:
                        straight_lines += 1
                        # No drawing here
                    else:
                        curved_lines += 1
                        classification = 'curved'
                        # No drawing here

            # Store line data with coordinates relative to ROI
            lines_data.append((x1, y1, x2, y2, classification))

        # --- Update counts and ratio based on ROI analysis ---
        self.straight_line_count = straight_lines
        self.curved_line_count = curved_lines
        total_lines = straight_lines + curved_lines
        self.line_ratio = straight_lines / total_lines if total_lines > 0 else 0.0

        # --- Update plotter data if enabled ---
        current_time = time.time()
        if self.plotter and self.show_charts and (current_time - self.last_plot_update_time > self.plot_update_interval):
             self.plotter.add_data_point(
                 self.straight_line_count, self.curved_line_count, self.line_ratio
             )
             self.last_plot_update_time = current_time

        # Return counts and the detailed line data (ROI coordinates)
        return straight_lines, curved_lines, lines_data

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

        # Ensure start point is within bounds before loop (important!)
        if not (0 <= x_curr < w and 0 <= y_curr < h):
            # If the start point given by Hough is somehow outside the ROI bounds,
            # it's likely an edge case or artifact. Return empty.
            # print(f"Warning: Start point ({x1},{y1}) outside edges bounds ({w}x{h})")
            return []


        while True:
            # Check bounds *inside* the loop as well
            if 0 <= x_curr < w and 0 <= y_curr < h:
                 # Check if the pixel is an edge point
                 if edges[y_curr, x_curr] > 0:
                     points.append((x_curr, y_curr))
            else:
                 # If we step outside bounds during iteration, stop collecting for this line
                 # print(f"Warning: Stepped outside edges bounds at ({x_curr},{y_curr})")
                 break # Or maybe continue if one coord is still valid? Bresenham handles this.


            if x_curr == x2 and y_curr == y2: break # Reached end point

            # Bresenham algorithm steps
            e2 = 2 * err
            if e2 >= dy: # Error threshold crossed for x
                if x_curr == x2: break # Prevent overshooting if x matches end point
                err += dy
                x_curr += sx
            if e2 <= dx: # Error threshold crossed for y
                if y_curr == y2: break # Prevent overshooting if y matches end point
                err += dx
                y_curr += sy

        return points


    def calculate_curvature(self, points):
        """
        Calculates curvature based on the deviation of the MIDDLE point(s)
        from the straight line connecting endpoints, normalized by segment length.
        Points are expected to be relative to the image they were extracted from (ROI).
        """
        if len(points) < 3:
            return 0.0 # Cannot calculate curvature

        p_start = np.array(points[0])
        p_end = np.array(points[-1])
        line_vec = p_end - p_start
        segment_length = np.linalg.norm(line_vec)

        if segment_length < 1e-6: # Avoid division by zero for coincident points
            return 0.0

        # --- Calculate deviation of the middle point ---
        middle_index = len(points) // 2
        # This check should be redundant if len(points) >= 3, but safe to keep
        if middle_index == 0 or middle_index >= len(points) -1:
             # This case really shouldn't happen with len >= 3.
             # Could indicate an issue in get_line_points if it occurs.
             print(f"Warning: Invalid middle_index {middle_index} for points length {len(points)}")
             return 0.0


        p_middle = np.array(points[middle_index])
        point_vec = p_middle - p_start # Vector from start to middle point

        # Calculate perpendicular distance using dot product projection
        # Project point_vec onto line_vec
        # proj_length = np.dot(point_vec, line_vec) / segment_length**2 # Incorrect, needs normalized vec
        # Use normalized line vector for projection length calculation
        line_vec_norm = line_vec / segment_length
        proj_length_on_line = np.dot(point_vec, line_vec_norm)

        # Find the closest point on the infinite line defined by start/end
        closest_point_on_line = p_start + line_vec_norm * proj_length_on_line

        # Calculate the distance vector (perpendicular)
        distance_vec = p_middle - closest_point_on_line
        distance = np.linalg.norm(distance_vec)


        # --- Old cross-product method (kept for reference, should give same result for 2D) ---
        # line_vec_norm = line_vec / segment_length # Normalized direction vector
        # Calculate perpendicular distance using 2D cross product magnitude
        # | V x W | = | Vx*Wy - Vy*Wx | for V=(Vx,Vy), W=(Wx,Wy)
        # distance = abs(point_vec[0] * line_vec_norm[1] - point_vec[1] * line_vec_norm[0])
        # --- End Old Method ---


        # Normalize the distance by the segment length
        curvature = distance / segment_length
        return curvature

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

        # Ensure minimum size (e.g., 10x10)
        min_roi_w = 10
        min_roi_h = 10
        self.roi_w = max(min_roi_w, self.roi_w)
        self.roi_h = max(min_roi_h, self.roi_h)

        # Constrain position and size
        self.roi_x = max(0, self.roi_x)
        self.roi_y = max(0, self.roi_y)
        # Adjust width/height if they exceed bounds based on current x/y
        self.roi_w = min(self.roi_w, frame_w - self.roi_x)
        self.roi_h = min(self.roi_h, frame_h - self.roi_y)
        # Re-check position just in case width/height adjustment caused issues (shouldn't usually)
        self.roi_x = min(self.roi_x, frame_w - self.roi_w)
        self.roi_y = min(self.roi_y, frame_h - self.roi_h)

        # Ensure width/height didn't become smaller than minimum due to constraint
        self.roi_w = max(min_roi_w, self.roi_w)
        self.roi_h = max(min_roi_h, self.roi_h)
        # Final check if min size pushes it out of bounds again (can happen at edges)
        self.roi_x = min(self.roi_x, frame_w - self.roi_w)
        self.roi_y = min(self.roi_y, frame_h - self.roi_h)




    def process_frame(self, frame):
        """Apply selected processing to the frame, using ROI for analysis."""
        # Ensure ROI is valid before processing
        if self.roi_enabled:
            self._ensure_roi_bounds() # Make sure ROI is within frame

        processed_frame = frame.copy()
        if self.flip_horizontal: processed_frame = cv2.flip(processed_frame, 1)
        if self.flip_vertical: processed_frame = cv2.flip(processed_frame, 0)

        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        analysis_source = gray # Start with grayscale
        if self.blur: # Apply blur BEFORE potential cropping if enabled
            analysis_source = cv2.GaussianBlur(gray, (15, 15), 0) # Consider kernel size based on ROI?

        # --- ROI Handling ---
        lines_data_roi = [] # Store results from analysis
        edges_roi = None
        if self.analyze_lines and self.roi_enabled and self.roi_w > 0 and self.roi_h > 0:
            # Crop the source image for analysis (blurred or just gray)
            roi_to_analyze = analysis_source[self.roi_y : self.roi_y + self.roi_h,
                                             self.roi_x : self.roi_x + self.roi_w]

            # Perform Canny edge detection *only on the ROI*
            edges_roi = cv2.Canny(roi_to_analyze, self.canny_threshold1, self.canny_threshold2)

            # Analyze lines within the edges_roi
            # This function now updates self.straight_line_count, self.curved_line_count, self.line_ratio
            # and returns line data with coordinates relative to the ROI.
            _, _, lines_data_roi = self.analyze_edge_lines(edges_roi)
        elif self.analyze_lines: # Analysis enabled but ROI is not (or invalid size)
            # Reset counts if analysis is expected but not performed
            self.straight_line_count = 0
            self.curved_line_count = 0
            self.line_ratio = 0.0
            # Optionally, could run analysis on the whole frame here as a fallback:
            # edges_full = cv2.Canny(analysis_source, ...)
            # _, _, lines_data_roi = self.analyze_edge_lines(edges_full) # Coords would be global


        # --- Determine Base Display Frame ---
        display_frame = processed_frame # Default to processed color frame
        if self.grayscale:
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # Show grayscale

        if self.edge_detection:
            # If edge detection view is on, show the ROI edges if available,
            # otherwise show full frame edges (or blank if not computed)
            if edges_roi is not None:
                # Create a black canvas and place the ROI edges onto it
                full_edges_display = np.zeros_like(processed_frame) # Same size as original frame
                edges_roi_bgr = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)
                full_edges_display[self.roi_y : self.roi_y + self.roi_h,
                                   self.roi_x : self.roi_x + self.roi_w] = edges_roi_bgr
                display_frame = full_edges_display

                # Option: Overlay ROI edges on grayscale instead of black
                # display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # display_frame[self.roi_y : self.roi_y + self.roi_h,
                #               self.roi_x : self.roi_x + self.roi_w] = np.maximum(
                #                   display_frame[self.roi_y : self.roi_y + self.roi_h, self.roi_x : self.roi_x + self.roi_w],
                #                   edges_roi_bgr)

            else:
                 # Fallback: Show Canny edges on the whole frame if ROI wasn't processed
                 edges_full = cv2.Canny(analysis_source, self.canny_threshold1, self.canny_threshold2)
                 display_frame = cv2.cvtColor(edges_full, cv2.COLOR_GRAY2BGR)


        # --- Draw Lines (if analysis occurred and visualization enabled) ---
        if lines_data_roi: # Check if analysis produced line data
             draw_on_edge_view = self.edge_detection and self.show_line_analysis
             draw_on_processed_view = not self.edge_detection and self.show_lines_on_original

             if draw_on_edge_view or draw_on_processed_view:
                  for x1_roi, y1_roi, x2_roi, y2_roi, classification in lines_data_roi:
                       # Offset coordinates from ROI-local to full-frame global
                       x1_global = x1_roi + self.roi_x
                       y1_global = y1_roi + self.roi_y
                       x2_global = x2_roi + self.roi_x
                       y2_global = y2_roi + self.roi_y

                       color = (0, 255, 0) if classification == 'straight' else (0, 0, 255)
                       cv2.line(display_frame, (x1_global, y1_global), (x2_global, y2_global), color, 2)


        # --- Draw ROI Rectangle on the display frame ---
        if self.roi_enabled:
            cv2.rectangle(display_frame,
                          (self.roi_x, self.roi_y),
                          (self.roi_x + self.roi_w, self.roi_y + self.roi_h),
                          self.roi_color, 1) # Draw ROI box

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
            self.add_info_overlay(display_frame) # Pass the frame we're actually showing

        # --- Apply Scaling ---
        if self.scale_factor != 1.0:
             height, width = display_frame.shape[:2]
             # Prevent zero or negative dimensions
             new_width = max(10, int(width * self.scale_factor))
             new_height = max(10, int(height * self.scale_factor))
             if new_width > 0 and new_height > 0:
                  display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
             else:
                  print(f"Warning: Invalid resize dimensions ({new_width}x{new_height}), skipping resize.")


        # Return the original processed frame (for recording) and the final display frame
        return processed_frame, display_frame


    def add_info_overlay(self, frame_to_draw_on):
        """Adds text information overlay to the frame."""
        height, width = frame_to_draw_on.shape[:2]
        y_offset = 25; x_offset = 10; font_scale = 0.6
        font_face = cv2.FONT_HERSHEY_SIMPLEX; font_color = (0, 255, 0) # Green
        bg_color = (0,0,0); font_thickness = 1; line_spacing = 25

        # Helper to draw text with background
        def draw_text(img, text, pos):
            nonlocal y_offset
            text_size, _ = cv2.getTextSize(text, font_face, font_scale, font_thickness)
            text_w, text_h = text_size
            #cv2.rectangle(img, (pos[0]-2, pos[1]-text_h-2), (pos[0]+text_w+2, pos[1]+2), bg_color, -1) # Simple rect bg
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
            prefix = "(ROI) " if self.roi_enabled else "(Full) " # Indicate source of numbers
            draw_text(frame_to_draw_on, f"{prefix}Straight: {self.straight_line_count}", (x_offset, y_offset))
            draw_text(frame_to_draw_on, f"{prefix}Curved: {self.curved_line_count}", (x_offset, y_offset))
            draw_text(frame_to_draw_on, f"{prefix}Ratio S/T: {self.line_ratio:.3f}", (x_offset, y_offset))
            draw_text(frame_to_draw_on, f"Curv Thresh: {self.curvature_threshold:.3f}", (x_offset, y_offset))
            draw_text(frame_to_draw_on, f"Canny: {self.canny_threshold1}/{self.canny_threshold2}", (x_offset, y_offset))
            draw_text(frame_to_draw_on, f"Hough MinLen: {self.hough_min_line_length}", (x_offset, y_offset))
            draw_text(frame_to_draw_on, f"Hough MaxGap: {self.hough_max_line_gap}", (x_offset, y_offset))


        # Recording Indicator (Top Right)
        if self.record:
            rec_text = "REC"
            text_size, _ = cv2.getTextSize(rec_text, font_face, 0.5, 1)
            text_w, text_h = text_size
            rec_center_x = width - text_w - 15 # Position based on text width
            rec_center_y = 25
            cv2.circle(frame_to_draw_on, (rec_center_x + text_w // 2, rec_center_y - text_h // 4), 8, (0, 0, 255), -1) # Circle near text
            cv2.putText(frame_to_draw_on, rec_text, (rec_center_x, rec_center_y), font_face, 0.5,(0, 0, 255), 1, cv2.LINE_AA)



    def run(self):
        """Run the main camera stream loop."""
        if not self.start(): return

        # Create window here to allow interaction before first frame
        cv2.namedWindow(self.window_name)

        # --- Matplotlib Event Loop Integration ---
        # Try to run matplotlib's event loop periodically if charts are shown
        # This is tricky and backend-dependent. Using plt.pause() is often simplest.
        last_plt_pause_time = time.time()
        plt_pause_interval = 0.05 # Pause duration in seconds

        try:
            while True:
                # --- Check for Matplotlib events/updates if charts are active ---
                if self.show_charts and self.plotter and self.plotter.fig.canvas:
                     current_time = time.time()
                     if current_time - last_plt_pause_time > plt_pause_interval:
                         plt.pause(0.001) # Very short pause allows GUI events processing
                         last_plt_pause_time = current_time
                     # Alternative: Explicitly flush events (can be backend specific)
                     # try:
                     #     self.plotter.fig.canvas.flush_events()
                     # except NotImplementedError:
                     #     pass # Fallback to pause

                # --- Check if plotter window was closed by user ---
                if self.plotter and not plt.fignum_exists(self.plotter.fig.number):
                     print("Plotter window closed by user. Disabling charts.")
                     self.show_charts = False
                     self.plotter = None # Allow re-creation if 'c' is pressed again


                # --- Read Camera Frame ---
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Warning: Failed to capture frame.")
                    # Check if camera physically disconnected
                    if not self.cap.isOpened():
                         print("Camera disconnected. Exiting.")
                         break
                    time.sleep(0.05) # Wait a bit before retrying
                    continue

                # --- Process Frame (including ROI analysis, drawing, etc.) ---
                original_processed_frame, display_frame = self.process_frame(frame)

                # --- Handle Recording ---
                if self.record:
                    if self.writer is None: # Start recording
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_filename = f"camera_rec_{timestamp}.avi"
                        h, w = original_processed_frame.shape[:2] # Use processed frame dims
                        if w > 0 and h > 0:
                            try:
                                rec_fps = min(self.fps, 30.0) if self.fps > 0 else 20.0 # Use measured FPS or default
                                rec_fps = max(5.0, rec_fps) # Ensure minimum sensible FPS
                                self.writer = cv2.VideoWriter(output_filename, fourcc, rec_fps, (w, h))
                                if not self.writer.isOpened(): raise IOError("VideoWriter failed to open")
                                print(f"Recording started: {output_filename} at ~{rec_fps:.1f} FPS")
                            except Exception as e:
                                print(f"Error initializing VideoWriter: {e}")
                                self.record = False # Turn off flag if writer fails
                                if self.writer: self.writer.release(); self.writer = None
                        else:
                            print("Error: Cannot start recording with invalid frame dimensions.")
                            self.record = False

                    if self.writer is not None and self.writer.isOpened():
                         try:
                             self.writer.write(original_processed_frame) # Write the frame used for processing
                         except Exception as e:
                             print(f"Error writing frame to video: {e}")
                             # Optionally stop recording on write error
                             # self.record = False
                elif not self.record and self.writer is not None: # Stop recording
                    print("Stopping recording...")
                    self.writer.release()
                    self.writer = None
                    print("Recording stopped.")

                # --- Display Frame ---
                cv2.imshow(self.window_name, display_frame)

                # --- Handle Keyboard Input ---
                # Reduce waitKey time for better responsiveness, esp with matplotlib GUI
                key = cv2.waitKey(1) & 0xFF # Use 1ms wait

                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key != 0xFF: # Check if a key was actually pressed
                    self.handle_key_press(key, original_processed_frame) # Pass frame for snapshot

        finally:
            print("Cleaning up...")
            if self.cap is not None:
                self.cap.release()
                print("Camera released.")
            if self.writer is not None:
                if self.writer.isOpened():
                     print("Releasing active video writer...")
                     self.writer.release()
                print("Video writer released.")

            # Close plotter window properly
            if self.plotter is not None:
                 self.plotter.close() # Use the plotter's close method
                 self.plotter = None

            cv2.destroyAllWindows()
            print("OpenCV windows destroyed.")
            # Ensure matplotlib interactive mode is off if it was turned on
            # plt.ioff() # Usually not necessary if closing windows properly


    def handle_key_press(self, key, frame_for_snapshot):
        """Handles key presses for changing settings and ROI."""
        # ROI Adjustment Flags
        adjust_roi_pos = False
        adjust_roi_size = False

        # --- Basic Toggles ---
        if key == ord('g'): self.grayscale = not self.grayscale; print(f"Grayscale: {self.grayscale}")
        elif key == ord('h'): self.flip_horizontal = not self.flip_horizontal; print(f"H-Flip: {self.flip_horizontal}")
        elif key == ord('v'): self.flip_vertical = not self.flip_vertical; print(f"V-Flip: {self.flip_vertical}")
        elif key == ord('b'): self.blur = not self.blur; print(f"Blur: {self.blur}")
        elif key == ord('e'): self.edge_detection = not self.edge_detection; print(f"Edge View: {self.edge_detection}")
        elif key == ord('i'): self.display_info = not self.display_info; print(f"Info: {self.display_info}")
        elif key == ord('l'):
            self.analyze_lines = not self.analyze_lines; print(f"Line Analysis: {self.analyze_lines}")
            if not self.analyze_lines:
                 self.show_line_analysis = False; print("Line Vis (Edge): Off")
                 self.show_lines_on_original = False; print("Line Vis (Orig): Off")
                 # Reset counts when turning off analysis
                 self.straight_line_count = 0
                 self.curved_line_count = 0
                 self.line_ratio = 0.0
        elif key == ord('a'): # Toggle line vis on edge view
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
                if self.plotter is None:
                    print("Initializing plotter...")
                    if not plt.isinteractive(): plt.ion() # Ensure interactive mode
                    self.plotter = DataPlotter()
                    self.last_plot_update_time = time.time() # Reset plot update timer
                    # Redraw the plot immediately if possible
                    # self.plotter.update_plot(0) # Might cause issues if called outside anim loop
                    try:
                        # Force redraw/update
                        self.plotter.fig.canvas.draw_idle()
                        self.plotter.fig.canvas.flush_events()
                    except Exception as e: print(f"Plotter init draw error: {e}")

                if not self.analyze_lines:
                     self.analyze_lines = True; print("Line analysis auto ON for charts.")
                # Add a point immediately to populate charts if data exists
                self.plotter.add_data_point(self.straight_line_count, self.curved_line_count, self.line_ratio)
                print("Charts: On")
            else:
                print("Charts: Off")
                # Don't destroy plotter, just hide/stop updating? Closing is better.
                if self.plotter:
                     self.plotter.close()
                     self.plotter = None

        elif key == ord('r'): self.record = not self.record # Messages handled in run loop
        elif key == ord('s'): # Snapshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"camera_snapshot_{timestamp}.jpg"
            # Save the frame BEFORE overlays and scaling, but AFTER flips/blur etc.
            cv2.imwrite(filename, frame_for_snapshot)
            print(f"Snapshot saved: {filename}")
        elif key == ord('R'): # Toggle ROI analysis enable/disable
             self.roi_enabled = not self.roi_enabled
             print(f"ROI Analysis Enabled: {self.roi_enabled}")
             if not self.roi_enabled:
                 # Reset ROI-specific counts when disabling ROI analysis
                 # (Keep this behaviour or let them persist?)
                 self.straight_line_count = 0
                 self.curved_line_count = 0
                 self.line_ratio = 0.0


        # --- Window Scaling ---
        elif key == ord('+') or key == ord('='): self.scale_factor = min(3.0, self.scale_factor + 0.1); print(f"Scale: {self.scale_factor:.1f}x")
        elif key == ord('-') or key == ord('_'): self.scale_factor = max(0.2, self.scale_factor - 0.1); print(f"Scale: {self.scale_factor:.1f}x")

        # --- Canny Tuning ---
        elif key == ord('['): self.canny_threshold1 = max(0, self.canny_threshold1 - 10); self.canny_threshold2 = max(self.canny_threshold1 + 10, self.canny_threshold2); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")
        elif key == ord(']'): self.canny_threshold1 = min(245, self.canny_threshold1 + 10); self.canny_threshold2 = max(self.canny_threshold1 + 10, self.canny_threshold2); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")
        elif key == ord('{'): self.canny_threshold2 = max(self.canny_threshold1 + 10, self.canny_threshold2 - 10); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")
        elif key == ord('}'): self.canny_threshold2 = min(255, self.canny_threshold2 + 10); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")

        # --- Curvature Tuning ---
        elif key == ord(',') or key == ord('<'): self.curvature_threshold = max(0.001, self.curvature_threshold - 0.002); print(f"Curv Thresh: {self.curvature_threshold:.3f}")
        elif key == ord('.') or key == ord('>'): self.curvature_threshold += 0.002; print(f"Curv Thresh: {self.curvature_threshold:.3f}")

        # --- Hough Tuning ---
        elif key == ord('n'): self.hough_min_line_length = max(1, self.hough_min_line_length - 5); print(f"Hough MinLen: {self.hough_min_line_length}")
        elif key == ord('m'): self.hough_min_line_length += 5; print(f"Hough MinLen: {self.hough_min_line_length}")
        elif key == ord('j'): self.hough_max_line_gap = max(0, self.hough_max_line_gap - 2); print(f"Hough MaxGap: {self.hough_max_line_gap}")
        elif key == ord('k'): self.hough_max_line_gap += 2; print(f"Hough MaxGap: {self.hough_max_line_gap}")

        # --- ROI Controls (Arrows / WASD / Shift) ---
        # Need to check ROI dimensions exist
        if self.roi_x is not None:
             # Size Adjustment (Shift + Keys)
             if key == ord('W'): # Shift + W -> Decrease Height (faster)
                  self.roi_h -= 10; adjust_roi_size = True; print("ROI Resize")
             elif key == ord('S'): # Shift + S -> Increase Height (faster)
                  self.roi_h += 10; adjust_roi_size = True; print("ROI Resize")
             elif key == ord('A'): # Shift + A -> Decrease Width (faster)
                  self.roi_w -= 10; adjust_roi_size = True; print("ROI Resize")
             elif key == ord('D'): # Shift + D -> Increase Width (faster)
                  self.roi_w += 10; adjust_roi_size = True; print("ROI Resize")
             # Fine Size Adjustment (Shift + wasd - lowercase check if CapsLock is off)
             elif key == ord('w'): # shift + w -> Decrease Height (slower)
                 self.roi_h -= 2; adjust_roi_size = True; print("ROI Resize")
             elif key == ord('s') and key != ord('s'): # Make sure it's not the snapshot 's'
                 # This logic is tricky with waitKey. Usually Shift detection requires OS level hooks.
                 # A common workaround is to use different keys entirely or check key codes more precisely.
                 # For simplicity, let's use Shift+Arrows for coarse, WASD for fine position.
                 # Let's re-assign WASD to fine position, Arrows to coarse position.
                 pass # WASD for fine position below
             elif key == ord('a') and key != ord('a'): # Make sure it's not line vis 'a'
                 pass
             elif key == ord('d'):
                 self.roi_w += 2; adjust_roi_size = True; print("ROI Resize")


             # Position Adjustment (Arrows for Coarse, WASD for Fine)
             # OpenCV waitKey doesn't directly give arrow keys easily across OSs.
             # Using common alternatives like WASD or specific key codes if needed.
             # Let's assume WASD for fine, maybe 8/4/6/2 numpad for coarse if available?
             # Sticking to WASD for fine move, requires user to check if other keys pressed.
             # Let's try simple WASD for movement for now. Add arrows later if needed/possible.

             step_fine = 1
             step_coarse = 5 # Let's use Arrows for coarse move if possible

             # Fine Movement (WASD)
             if key == ord('w'): self.roi_y -= step_fine; adjust_roi_pos = True
             elif key == ord('s'): self.roi_y += step_fine; adjust_roi_pos = True
             elif key == ord('a'): self.roi_x -= step_fine; adjust_roi_pos = True
             elif key == ord('d'): self.roi_x += step_fine; adjust_roi_pos = True

             # Coarse Movement / Resize (Using placeholder keys, map arrows if needed)
             # Check common key codes (these might vary!)
             # 82: Up, 84: Down, 81: Left, 83: Right (Often on Linux/Windows)
             # Mac might use different codes (e.g., 126, 125, 123, 124)

             # Placeholder: Use Shift + WASD for Resize, Arrows for coarse move
             # For simplicity, let's map coarse move to IJKL keys for now
             if key == ord('i'): self.roi_y -= step_coarse; adjust_roi_pos = True
             elif key == ord('k'): self.roi_y += step_coarse; adjust_roi_pos = True
             elif key == ord('j'): self.roi_x -= step_coarse; adjust_roi_pos = True
             elif key == ord('l') and key != ord('l'): # Avoid conflict with analysis toggle
                 pass # Reassign 'l' conflict
                 # Let's use 'p' for coarse right move instead
             elif key == ord('p'):
                 self.roi_x += step_coarse; adjust_roi_pos = True


             # Resize using Shift + IJKL (mapping from Shift+Arrows idea)
             if key == ord('I'): # Shift + i -> Decrease Height Coarse
                 self.roi_h -= 10; adjust_roi_size = True; print("ROI Resize")
             elif key == ord('K'): # Shift + k -> Increase Height Coarse
                 self.roi_h += 10; adjust_roi_size = True; print("ROI Resize")
             elif key == ord('J'): # Shift + j -> Decrease Width Coarse
                 self.roi_w -= 10; adjust_roi_size = True; print("ROI Resize")
             elif key == ord('L') and key != ord('L'): # Shift + l
                  pass # Reassign 'L' conflict
                  # Use Shift + P for Coarse width increase
             elif key == ord('P'):
                  self.roi_w += 10; adjust_roi_size = True; print("ROI Resize")


             # If ROI was adjusted, ensure it stays within bounds
             if adjust_roi_pos or adjust_roi_size:
                  print(f"ROI Adjust: Pos={adjust_roi_pos}, Size={adjust_roi_size}")
                  self._ensure_roi_bounds()
                  print(f"New ROI: x={self.roi_x}, y={self.roi_y}, w={self.roi_w}, h={self.roi_h}")
                  # Reset analysis counts when ROI changes? Optional.
                  # self.straight_line_count = 0
                  # self.curved_line_count = 0
                  # self.line_ratio = 0.0


# --- Helper Function ---
def list_available_cameras(max_check=10):
    """List available camera indices."""
    available_cameras = []
    print(f"Checking for cameras up to index {max_check-1}...")
    for i in range(max_check):
        cap_test = cv2.VideoCapture(i)
        # Some systems might need a slight delay or a read attempt
        time.sleep(0.05)
        is_opened = cap_test.isOpened()
        if is_opened:
            # Try reading a frame as a more robust check
            ret, _ = cap_test.read()
            if ret:
                 available_cameras.append(i)
            else:
                 print(f"Index {i} opened but failed to read frame.")
        if cap_test is not None: # Ensure release happens even if not opened
            cap_test.release()
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