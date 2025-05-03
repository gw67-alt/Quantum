# Full code with modifications aimed at improving curved line detection robustness
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
        self.fig.suptitle('Camera Stream Analysis', fontsize=16)

        self.straight_line_plot, = self.axes[0].plot([], [], 'g-', label='Straight Lines')
        self.curved_line_plot, = self.axes[0].plot([], [], 'r-', label='Curved Lines')
        self.ratio_plot, = self.axes[1].plot([], [], 'b-', label='Straight/Total Ratio')
        self.total_plot, = self.axes[2].plot([], [], 'w-', label='Total Lines')

        self.axes[0].set_title('Line Counts')
        self.axes[0].set_ylabel('Count')
        self.axes[0].legend(loc='upper left')
        self.axes[0].grid(True, linestyle='--', alpha=0.6)

        self.axes[1].set_title('Line Ratio (Straight / Total)')
        self.axes[1].set_ylabel('Ratio')
        self.axes[1].set_ylim(0, 1.05)
        self.axes[1].legend(loc='upper left')
        self.axes[1].grid(True, linestyle='--', alpha=0.6)

        self.axes[2].set_title('Total Lines (Straight + Curved)')
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

        if not plt.isinteractive(): plt.ion()
        plt.show(block=False)
        self.fig.canvas.flush_events()
        self.start_time = time.time()

    def add_data_point(self, straight_lines, curved_lines, line_ratio):
        """Add a new data point to the plot queues."""
        with self.lock:
            current_time = time.time() - self.start_time
            if not self.timestamps or current_time > self.timestamps[-1]:
                self.timestamps.append(current_time)
                self.straight_lines.append(straight_lines)
                self.curved_lines.append(curved_lines)
                self.line_ratios.append(line_ratio)

    def update_plot(self, frame):
        """Update the plot with new data. Called by FuncAnimation."""
        with self.lock:
            if len(self.timestamps) <= 1 and self.timestamps[0] == 0:
                return self.straight_line_plot, self.curved_line_plot, self.ratio_plot, self.total_plot

            times_list = list(self.timestamps)
            straight_list = list(self.straight_lines)
            curved_list = list(self.curved_lines)
            ratio_list = list(self.line_ratios)

            if len(times_list) > 1:
                times_plot = times_list[1:]
                straight_plot = straight_list[1:]
                curved_plot = curved_list[1:]
                ratio_plot_data = ratio_list[1:]
            else:
                return self.straight_line_plot, self.curved_line_plot, self.ratio_plot, self.total_plot

        if not times_plot:
            return self.straight_line_plot, self.curved_line_plot, self.ratio_plot, self.total_plot

        #self.straight_line_plot.set_data(times_plot, straight_plot)
        self.curved_line_plot.set_data(times_plot, curved_plot)
        self.ratio_plot.set_data(times_plot, ratio_plot_data)
        total_lines = [s + c for s, c in zip(straight_plot, curved_plot)]
        self.total_plot.set_data(times_plot, total_lines)

        history_duration = 20.0
        current_max_time = times_plot[-1]
        x_min = max(0, current_max_time - history_duration)
        x_max = max(x_min + 10.0, current_max_time + 2.0)

        visible_indices = [i for i, t in enumerate(times_plot) if t >= x_min]
        if not visible_indices:
            y_max_lines = 10
            y_max_total = 10
        else:
            first_visible_index = visible_indices[0]
            visible_straight = straight_plot[first_visible_index:]
            visible_curved = curved_plot[first_visible_index:]
            visible_total = total_lines[first_visible_index:]
            max_s = max(visible_straight, default=0)
            max_c = max(visible_curved, default=0)
            max_t = max(visible_total, default=0)
            y_max_lines = max(max_s, max_c, 10)
            y_max_total = max(max_t, 10)

        self.axes[0].set_xlim(x_min, x_max)
        self.axes[0].set_ylim(0, y_max_lines * 1.2)
        self.axes[1].set_xlim(x_min, x_max)
        self.axes[2].set_xlim(x_min, x_max)
        self.axes[2].set_ylim(0, y_max_total * 1.2)

        return self.straight_line_plot, self.curved_line_plot, self.ratio_plot, self.total_plot

    def close(self):
        """Close the plot window."""
        plt.close(self.fig)
        print("Plotter window closed.")

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
        self.show_line_analysis = False

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
        self.window_name = "Camera Stream"
        self.scale_factor = 1.0

        # Data visualization
        self.show_charts = False
        self.plotter = None
        self.last_plot_update_time = 0
        self.plot_update_interval = 0.2 # Update plot data every 0.2s

    def start(self):
        """Start the camera stream."""
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

        print(f"Camera opened: Index {self.camera_index} | Res: {width}x{height} | Reported FPS: {cam_fps:.2f}")
        print("\n--- Controls ---")
        print(" q       - Quit")
        print(" g       - Toggle Grayscale")
        print(" h       - Flip Horizontal")
        print(" v       - Flip Vertical")
        print(" b       - Toggle Gaussian Blur (RECOMMENDED for line analysis)")
        print(" e       - Toggle Edge Detection View (Canny)")
        print(" l       - Toggle Line Analysis (Calculates straight/curved)")
        print(" a       - Toggle Line Analysis Visualization (on Edge View)")
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
        print("----------------")

        self.last_fps_update_time = time.time()
        self.accumulated_frames = 0
        return True

    def analyze_edge_lines(self, edges):
        """ Analyze edges using HoughLinesP and classify lines based on curvature."""
        vis_image = None
        if self.edge_detection and self.show_line_analysis:
             vis_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Use tunable Hough parameters
        lines = cv2.HoughLinesP(
            edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            # Using instance variables now
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        straight_lines = 0
        curved_lines = 0

        if lines is None:
            self.straight_line_count = 0
            self.curved_line_count = 0
            self.line_ratio = 0.0
            return vis_image if vis_image is not None else cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length_sq = (x2 - x1)**2 + (y2 - y1)**2

            # Skip curvature check for very short segments (treat as straight)
            if line_length_sq < 5**2: # Consider making this threshold relative or tunable
                 straight_lines += 1
                 if vis_image is not None:
                     cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                 continue

            points = self.get_line_points(edges, x1, y1, x2, y2)

            if len(points) < 3: # Need > 2 points for curvature check
                straight_lines += 1
                if vis_image is not None:
                    cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                continue

            # Calculate curvature (using potentially modified method)
            curvature = self.calculate_curvature(points)

            if curvature < self.curvature_threshold:
                straight_lines += 1
                if vis_image is not None:
                    cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                curved_lines += 1
                if vis_image is not None:
                    cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red for curved

        self.straight_line_count = straight_lines
        self.curved_line_count = curved_lines
        total_lines = straight_lines + curved_lines
        self.line_ratio = straight_lines / total_lines if total_lines > 0 else 0.0

        current_time = time.time()
        if self.plotter and self.show_charts and (current_time - self.last_plot_update_time > self.plot_update_interval):
             self.plotter.add_data_point(
                 self.straight_line_count, self.curved_line_count, self.line_ratio
             )
             self.last_plot_update_time = current_time

        return vis_image if vis_image is not None else cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def get_line_points(self, edges, x1, y1, x2, y2):
        """Extracts edge points using Bresenham's line algorithm principle."""
        points = []
        h, w = edges.shape[:2]
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        x_curr, y_curr = x1, y1

        while True:
            if 0 <= x_curr < w and 0 <= y_curr < h and edges[y_curr, x_curr] > 0:
                points.append((x_curr, y_curr))
            if x_curr == x2 and y_curr == y2: break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x_curr += sx
            if e2 <= dx:
                err += dx
                y_curr += sy
        return points

    def calculate_curvature(self, points):
        """
        Calculates curvature based on the deviation of the MIDDLE point(s)
        from the straight line connecting endpoints, normalized by segment length.
        This version is less sensitive to noisy endpoints than using max deviation.
        """
        if len(points) < 3:
            return 0.0 # Cannot calculate curvature

        p_start = np.array(points[0])
        p_end = np.array(points[-1])
        line_vec = p_end - p_start
        segment_length = np.linalg.norm(line_vec)

        if segment_length < 1e-6: # Avoid division by zero for coincident points
            return 0.0

        line_vec_norm = line_vec / segment_length # Normalized direction vector

        # --- Calculate deviation of the middle point ---
        middle_index = len(points) // 2
        # Ensure middle index isn't the start or end index itself
        # (Should be guaranteed by len(points) < 3 check, but double-check)
        if middle_index == 0 or middle_index >= len(points) -1:
             return 0.0 # Should not happen if len >= 3

        p_middle = np.array(points[middle_index])
        point_vec = p_middle - p_start # Vector from start to middle point

        # Calculate perpendicular distance using 2D cross product magnitude
        # | V x W | = | Vx*Wy - Vy*Wx | for V=(Vx,Vy), W=(Wx,Wy)
        distance = abs(point_vec[0] * line_vec_norm[1] - point_vec[1] * line_vec_norm[0])

        # Optional: Average with neighbors for more stability
        # if middle_index > 0 and middle_index < len(points) - 2:
        #    p_mid_prev = np.array(points[middle_index-1])
        #    p_mid_next = np.array(points[middle_index+1])
        #    # Calculate distances for neighbors and average
        #    # ... (more complex logic) ...

        # Normalize the distance by the segment length
        curvature = distance / segment_length
        return curvature

    def process_frame(self, frame):
        """Apply selected processing to the frame."""
        processed_frame = frame.copy()
        if self.flip_horizontal: processed_frame = cv2.flip(processed_frame, 1)
        if self.flip_vertical: processed_frame = cv2.flip(processed_frame, 0)

        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        analysis_gray = gray
        if self.blur: # Apply blur BEFORE Canny if enabled
             analysis_gray = cv2.GaussianBlur(gray, (15, 15), 0)

        edges = cv2.Canny(analysis_gray, self.canny_threshold1, self.canny_threshold2)

        line_vis_image = None
        if self.analyze_lines:
            line_vis_image = self.analyze_edge_lines(edges) # Updates counts/ratio

        display_frame = processed_frame
        if self.grayscale:
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self.edge_detection:
            if self.show_line_analysis and line_vis_image is not None:
                display_frame = line_vis_image
            else:
                display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        current_time = time.time()
        self.accumulated_frames += 1
        time_diff = current_time - self.last_fps_update_time
        if time_diff >= 1.0:
            self.fps = self.accumulated_frames / time_diff
            self.last_fps_update_time = current_time
            self.accumulated_frames = 0

        if self.display_info:
            self.add_info_overlay(display_frame)

        if self.scale_factor != 1.0:
             height, width = display_frame.shape[:2]
             new_width = max(10, int(width * self.scale_factor))
             new_height = max(10, int(height * self.scale_factor))
             display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return processed_frame, display_frame

    def add_info_overlay(self, frame_to_draw_on):
         """Adds text information overlay to the frame."""
         height, width = frame_to_draw_on.shape[:2]
         y_offset = 25; x_offset = 10; font_scale = 0.6
         font_face = cv2.FONT_HERSHEY_SIMPLEX; font_color = (0, 255, 0)
         font_thickness = 1; line_spacing = 25

         # Basic Info
         cv2.putText(frame_to_draw_on, f"FPS: {self.fps:.1f}", (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing
         cv2.putText(frame_to_draw_on, f"Res: {self.original_dimensions[0]}x{self.original_dimensions[1]}", (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing

         # Effects Status
         effects = []
         if self.grayscale: effects.append("Gray")
         if self.flip_horizontal: effects.append("HFlip")
         if self.flip_vertical: effects.append("VFlip")
         if self.blur: effects.append("Blur")
         if self.edge_detection: effects.append("Edges")
         if self.analyze_lines: effects.append("LineAn")
         if self.show_line_analysis: effects.append("LineVis")
         if self.show_charts: effects.append("Charts")
         if effects: cv2.putText(frame_to_draw_on, "FX: " + ", ".join(effects), (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing

         # Line Analysis Metrics & Parameters
         if self.analyze_lines:
             cv2.putText(frame_to_draw_on, f"Straight: {self.straight_line_count}", (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing
             cv2.putText(frame_to_draw_on, f"Curved: {self.curved_line_count}", (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing
             cv2.putText(frame_to_draw_on, f"Ratio S/T: {self.line_ratio:.3f}", (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing
             cv2.putText(frame_to_draw_on, f"Curv Thresh: {self.curvature_threshold:.3f}", (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing
             cv2.putText(frame_to_draw_on, f"Canny: {self.canny_threshold1}/{self.canny_threshold2}", (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing
             # Display tunable Hough params
             cv2.putText(frame_to_draw_on, f"Hough MinLen: {self.hough_min_line_length}", (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing
             cv2.putText(frame_to_draw_on, f"Hough MaxGap: {self.hough_max_line_gap}", (x_offset, y_offset), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA); y_offset += line_spacing


         # Recording Indicator
         if self.record:
             rec_center = (width - 25, 25); rec_radius = 8
             cv2.circle(frame_to_draw_on, rec_center, rec_radius, (0, 0, 255), -1)
             cv2.putText(frame_to_draw_on,"REC",(width - 60, 30), font_face, 0.5,(0, 0, 255), 1, cv2.LINE_AA)

    def run(self):
        """Run the main camera stream loop."""
        if not self.start(): return

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Warning: Failed to capture frame.")
                    #time.sleep(0.5)
                    if not self.cap.isOpened(): print("Camera disconnected. Exiting."); break
                    continue

                original_processed_frame, display_frame = self.process_frame(frame)

                if self.record:
                    if self.writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_filename = f"camera_rec_{timestamp}.avi"
                        h, w = original_processed_frame.shape[:2]
                        try:
                            rec_fps = 20.0
                            self.writer = cv2.VideoWriter(output_filename, fourcc, rec_fps, (w, h))
                            if not self.writer.isOpened(): raise IOError("Writer failed to open")
                            print(f"Recording started: {output_filename} at {rec_fps} FPS")
                        except Exception as e:
                             print(f"Error initializing VideoWriter: {e}"); self.record = False
                    if self.writer is not None:
                        self.writer.write(original_processed_frame)
                elif not self.record and self.writer is not None:
                    print("Stopping recording..."); self.writer.release(); self.writer = None; print("Recording stopped.")

                cv2.imshow(self.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'): print("Quitting..."); break
                elif key != 0xFF:
                    self.handle_key_press(key, original_processed_frame)

                # Removed direct flush_events call here to prevent GIL errors

        finally:
            print("Cleaning up...")
            if self.cap is not None: self.cap.release(); print("Camera released.")
            if self.writer is not None:
                if self.writer.isOpened(): self.writer.release()
                print("Video writer released.")
            if self.plotter is not None:
                try:
                    if self.plotter.fig and self.plotter.fig.canvas:
                         plt.close(self.plotter.fig)
                         print("Plotter window closed via plt.close().")
                except Exception as e: print(f"Warning: Exception closing plot: {e}")
                self.plotter = None
            cv2.destroyAllWindows(); print("OpenCV windows destroyed.")

    def handle_key_press(self, key, frame_for_snapshot):
        """Handles key presses for changing settings."""
        # ... (Keep existing key handling for g, h, v, b, e, i, l, a, c, r, +/-, [/], {/}) ...
        if key == ord('g'): self.grayscale = not self.grayscale; print(f"Grayscale: {self.grayscale}")
        elif key == ord('h'): self.flip_horizontal = not self.flip_horizontal; print(f"H-Flip: {self.flip_horizontal}")
        elif key == ord('v'): self.flip_vertical = not self.flip_vertical; print(f"V-Flip: {self.flip_vertical}")
        elif key == ord('b'): self.blur = not self.blur; print(f"Blur: {self.blur}")
        elif key == ord('e'): self.edge_detection = not self.edge_detection; print(f"Edge View: {self.edge_detection}")
        elif key == ord('i'): self.display_info = not self.display_info; print(f"Info: {self.display_info}")
        elif key == ord('l'):
             self.analyze_lines = not self.analyze_lines; print(f"Line Analysis: {self.analyze_lines}")
             if not self.analyze_lines: self.show_line_analysis = False; print("Line Vis: Off")
        elif key == ord('a'):
             if self.analyze_lines: self.show_line_analysis = not self.show_line_analysis; print(f"Line Vis: {self.show_line_analysis}")
             else: print("Enable Line Analysis ('l') first.")
        elif key == ord('c'):
            self.show_charts = not self.show_charts
            if self.show_charts:
                if self.plotter is None:
                     print("Initializing plotter...");
                     if not plt.isinteractive(): plt.ion()
                     self.plotter = DataPlotter(); self.last_plot_update_time = time.time()
                if not self.analyze_lines: self.analyze_lines = True; print("Line analysis auto ON.")
                self.plotter.add_data_point(self.straight_line_count, self.curved_line_count, self.line_ratio)
                print("Charts: On")
            else: print("Charts: Off")
        elif key == ord('r'): self.record = not self.record # Messages handled in run loop
        elif key == ord('+') or key == ord('='): self.scale_factor = min(3.0, self.scale_factor + 0.1); print(f"Scale: {self.scale_factor:.1f}x")
        elif key == ord('-') or key == ord('_'): self.scale_factor = max(0.2, self.scale_factor - 0.1); print(f"Scale: {self.scale_factor:.1f}x")
        elif key == ord('['): self.canny_threshold1 = max(0, self.canny_threshold1 - 10); self.canny_threshold2 = max(self.canny_threshold1 + 10, self.canny_threshold2); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")
        elif key == ord(']'): self.canny_threshold1 += 10; self.canny_threshold2 = max(self.canny_threshold1 + 10, self.canny_threshold2); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")
        elif key == ord('{'): self.canny_threshold2 = max(self.canny_threshold1 + 10, self.canny_threshold2 - 10); print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")
        elif key == ord('}'): self.canny_threshold2 += 10; print(f"Canny: {self.canny_threshold1}/{self.canny_threshold2}")
        elif key == ord(',') or key == ord('<'): self.curvature_threshold = max(0.001, self.curvature_threshold - 0.002); print(f"Curv Thresh: {self.curvature_threshold:.3f}")
        elif key == ord('.') or key == ord('>'): self.curvature_threshold += 0.002; print(f"Curv Thresh: {self.curvature_threshold:.3f}")
        # -- New Keys for Hough Parameters --
        elif key == ord('n'): # Decrease Min Line Length
             self.hough_min_line_length = max(1, self.hough_min_line_length - 5) # Ensure min length >= 1
             print(f"Hough MinLen: {self.hough_min_line_length}")
        elif key == ord('m'): # Increase Min Line Length
             self.hough_min_line_length += 5
             print(f"Hough MinLen: {self.hough_min_line_length}")
        elif key == ord('j'): # Decrease Max Line Gap
             self.hough_max_line_gap = max(0, self.hough_max_line_gap - 2) # Ensure max gap >= 0
             print(f"Hough MaxGap: {self.hough_max_line_gap}")
        elif key == ord('k'): # Increase Max Line Gap
             self.hough_max_line_gap += 2
             print(f"Hough MaxGap: {self.hough_max_line_gap}")
        # -- Snapshot --
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"camera_snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame_for_snapshot) # Save frame after flips, before overlays
            print(f"Snapshot saved: {filename}")

# --- Helper Function ---
def list_available_cameras(max_check=10):
    """List available camera indices."""
    available_cameras = []
    print(f"Checking for cameras up to index {max_check-1}...")
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available_cameras.append(i)
            cap.release()
    # print(f"Found indices: {available_cameras}") # Already printed below if found
    return available_cameras

# --- Main Execution ---
if __name__ == "__main__":
    camera_index = 0
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            cameras = list_available_cameras()
            if not cameras: print("No cameras detected.")
            else: print(f"Available camera indices: {cameras}")
            sys.exit(0)
        else:
            try: camera_index = int(sys.argv[1])
            except ValueError: print(f"Error: Invalid camera index '{sys.argv[1]}'."); sys.exit(1)

    print(f"Attempting to use camera index: {camera_index}")
    stream = CameraStream(camera_index=camera_index)
    stream.run()
    print("Program finished.")
