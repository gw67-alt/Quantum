import cv2
import numpy as np
import time
import sys
import math

class CameraStream:
    def __init__(self, camera_index=0):
        """
        Initialize a camera stream with various display options.
        
        Parameters:
        -----------
        camera_index : int
            Index of the camera to use (default: 0 for primary webcam)
        """
        self.camera_index = camera_index
        self.cap = None
        self.frame_counter = 0
        self.start_time = 0
        self.fps = 0
        
        # Processing flags
        self.grayscale = False
        self.flip_horizontal = False
        self.flip_vertical = False
        self.blur = False
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
        
        # Edge detection parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # Line detection parameters
        self.hough_threshold = 50
        self.hough_min_line_length = 50
        self.hough_max_line_gap = 10
        self.curvature_threshold = 0.02  # Threshold for determining curved vs straight lines
        
        # Window settings
        self.window_name = "Camera Stream"
        self.scale_factor = 1.0

    def start(self):
        """Start the camera stream."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera with index {self.camera_index}")
            return False
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_dimensions = (width, height)
        
        print(f"Camera opened with resolution: {width}x{height}")
        print("\nControls:")
        print("  q     - Quit")
        print("  g     - Toggle grayscale")
        print("  h     - Flip horizontal")
        print("  v     - Flip vertical")
        print("  b     - Toggle blur")
        print("  e     - Toggle edge detection")
        print("  i     - Toggle info display")
        print("  r     - Toggle recording")
        print("  +/-   - Resize window")
        print("  s     - Save current frame")
        print("  l     - Toggle line analysis")
        print("  a     - Toggle line analysis visualization")
        print("  [/]   - Decrease/Increase Canny low threshold")
        print("  {/}   - Decrease/Increase Canny high threshold")
        print("  </,>  - Decrease/Increase curvature threshold")
        
        self.start_time = time.time()
        return True

    def analyze_edge_lines(self, edges):
        """
        Analyze edges to detect straight and curved lines.
        Returns the number of straight lines, curved lines, and a visualization image.
        """
        # Create a color image for visualization
        vis_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Use HoughLinesP to detect straight line segments
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.hough_threshold, 
            minLineLength=self.hough_min_line_length, 
            maxLineGap=self.hough_max_line_gap
        )
        
        straight_lines = 0
        curved_lines = 0
        
        # If no lines detected
        if lines is None:
            self.straight_line_count = 0
            self.curved_line_count = 0
            self.line_ratio = 0.0
            return vis_image
        
        # Process detected lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check for contour curvature (using intermediate points)
            # For a straight line, all points should lie on the line
            # Extract points along the edge between (x1,y1) and (x2,y2)
            points = self.get_line_points(edges, x1, y1, x2, y2)
            
            if len(points) < 3:  # Need at least 3 points to measure curvature
                straight_lines += 1
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for straight
                continue
                
            # Calculate curvature of the line
            curvature = self.calculate_curvature(points)
            
            if curvature < self.curvature_threshold:
                straight_lines += 1
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for straight
            else:
                curved_lines += 1
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for curved
                
        self.straight_line_count = straight_lines
        self.curved_line_count = curved_lines
        
        # Calculate the ratio of straight to total lines
        total_lines = straight_lines + curved_lines
        self.line_ratio = straight_lines / max(1, total_lines)
        
        return vis_image
    
    def get_line_points(self, edges, x1, y1, x2, y2):
        """
        Extract points along an edge between two endpoints.
        """
        points = []
        
        # Calculate line parameters
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steep = dy > dx
        
        # If the line is steep, transpose the image
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
            
        # Make sure x1 <= x2
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            
        # Calculate the Y step direction
        step_y = 1 if y1 < y2 else -1
        
        # Calculate error (distance to true line)
        dx = x2 - x1
        dy = abs(y2 - y1)
        error = dx // 2
        y = y1
        
        # Iterate through points along the line
        for x in range(x1, x2 + 1):
            # Add point if it's an edge pixel
            coord = (y, x) if steep else (x, y)
            if 0 <= coord[0] < edges.shape[1] and 0 <= coord[1] < edges.shape[0]:
                if edges[coord[1], coord[0]] > 0:
                    points.append(coord)
            
            # Update error and potentially step in y
            error -= dy
            if error < 0:
                y += step_y
                error += dx
                
        return points
    
    def calculate_curvature(self, points):
        """
        Calculate the curvature of a set of points.
        Higher values indicate more curved lines.
        """
        if len(points) < 3:
            return 0
            
        # Calculate distances from each point to the line connecting endpoints
        p1 = np.array(points[0])
        p2 = np.array(points[-1])
        
        # Skip if endpoints are identical
        if np.array_equal(p1, p2):
            return 0
            
        # Calculate the line vector
        line_vec = p2 - p1
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            return 0
            
        # Normalize line vector
        line_vec = line_vec / line_length
        
        # Calculate distances from each point to the line
        max_distance = 0
        for p in points[1:-1]:  # Skip endpoints
            # Vector from p1 to current point
            point_vec = np.array(p) - p1
            
            # Calculate projection of point on the line
            proj_length = np.dot(point_vec, line_vec)
            
            # Calculate perpendicular distance
            proj_point = p1 + proj_length * line_vec
            distance = np.linalg.norm(np.array(p) - proj_point)
            
            max_distance = max(max_distance, distance)
            
        # Normalize by the length of the line
        curvature = max_distance / max(1, line_length)
        return curvature

    def process_frame(self, frame):
        """Apply selected processing to the frame."""
        # Apply flips if enabled
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)  # 1 for horizontal flip
        if self.flip_vertical:
            frame = cv2.flip(frame, 0)  # 0 for vertical flip
            
        # Make a copy to avoid modifying the original for recording
        display_frame = frame.copy()
        
        # Create edge map for analysis
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply blur before edge detection if enabled
        if self.blur:
            gray = cv2.GaussianBlur(gray, (15, 15), 0)
            display_frame = cv2.GaussianBlur(display_frame, (15, 15), 0)
            
        # Detect edges for analysis or display
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        
        # Apply line analysis if enabled
        line_vis_image = None
        if self.analyze_lines:
            line_vis_image = self.analyze_edge_lines(edges)
        
        # Apply grayscale effect if enabled
        if self.grayscale:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            # Convert back to BGR for consistent display and text rendering
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            
        # Apply edge detection effect if enabled
        if self.edge_detection:
            if self.show_line_analysis and line_vis_image is not None:
                display_frame = line_vis_image
            else:
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                display_frame = edges_colored
            
        # Calculate and display FPS
        self.frame_counter += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            self.fps = self.frame_counter / elapsed_time
            self.frame_counter = 0
            self.start_time = time.time()
            
        if self.display_info:
            # Add info text
            height, width = display_frame.shape[:2]
            
            info_text = []
            info_text.append(f"FPS: {self.fps:.1f}")
            info_text.append(f"Resolution: {width}x{height}")
            
            # Add status indicators for enabled effects
            effects = []
            if self.grayscale:
                effects.append("Grayscale")
            if self.flip_horizontal:
                effects.append("H-Flip")
            if self.flip_vertical:
                effects.append("V-Flip")
            if self.blur:
                effects.append("Blur")
            if self.edge_detection:
                effects.append("Edges")
            if self.record:
                effects.append("REC")
            if self.analyze_lines:
                effects.append("Line Analysis")
                
            # Add line analysis metrics if enabled
            if self.analyze_lines:
                info_text.append(f"Straight Lines: {self.straight_line_count}")
                info_text.append(f"Curved Lines: {self.curved_line_count}")
                info_text.append(f"Straight/Total Ratio: {self.line_ratio:.3f}")
                info_text.append(f"Curvature Threshold: {self.curvature_threshold:.3f}")
                info_text.append(f"Canny Thresholds: {self.canny_threshold1}/{self.canny_threshold2}")
                
            # Add red recording indicator
            if self.record:
                # Draw a red circle in top-right corner
                cv2.circle(display_frame, (width-20, 20), 10, (0, 0, 255), -1)
                
            if effects:
                info_text.append("Effects: " + ", ".join(effects))
                
            # Display all text
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(
                    display_frame, 
                    text, 
                    (10, y_offset + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
        # Resize the frame based on scale factor
        if self.scale_factor != 1.0:
            width = int(width * self.scale_factor)
            height = int(height * self.scale_factor)
            display_frame = cv2.resize(display_frame, (width, height))
            
        return frame, display_frame  # Return both original and display frames

    def run(self):
        """Run the main camera stream loop."""
        if not self.start():
            return
            
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to capture frame. Exiting...")
                    break
                    
                original_frame, display_frame = self.process_frame(frame)
                
                # Record video if enabled
                if self.record and self.writer is None:
                    # Initialize video writer
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    output_filename = f"camera_recording_{timestamp}.avi"
                    h, w = original_frame.shape[:2]
                    self.writer = cv2.VideoWriter(
                        output_filename, 
                        fourcc, 
                        20.0,  # FPS 
                        (w, h)
                    )
                    print(f"Recording started: {output_filename}")
                    
                if self.record and self.writer is not None:
                    self.writer.write(original_frame)
                    
                # Display the frame
                cv2.imshow(self.window_name, display_frame)
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('g'):
                    self.grayscale = not self.grayscale
                    print(f"Grayscale: {'On' if self.grayscale else 'Off'}")
                elif key == ord('h'):
                    self.flip_horizontal = not self.flip_horizontal
                    print(f"Horizontal Flip: {'On' if self.flip_horizontal else 'Off'}")
                elif key == ord('v'):
                    self.flip_vertical = not self.flip_vertical
                    print(f"Vertical Flip: {'On' if self.flip_vertical else 'Off'}")
                elif key == ord('b'):
                    self.blur = not self.blur
                    print(f"Blur: {'On' if self.blur else 'Off'}")
                elif key == ord('e'):
                    self.edge_detection = not self.edge_detection
                    print(f"Edge Detection: {'On' if self.edge_detection else 'Off'}")
                elif key == ord('i'):
                    self.display_info = not self.display_info
                    print(f"Info Display: {'On' if self.display_info else 'Off'}")
                elif key == ord('l'):
                    self.analyze_lines = not self.analyze_lines
                    print(f"Line Analysis: {'On' if self.analyze_lines else 'Off'}")
                elif key == ord('a'):
                    self.show_line_analysis = not self.show_line_analysis
                    print(f"Line Analysis Visualization: {'On' if self.show_line_analysis else 'Off'}")
                elif key == ord('r'):
                    self.record = not self.record
                    if not self.record and self.writer is not None:
                        self.writer.release()
                        self.writer = None
                        print("Recording stopped")
                elif key == ord('+') or key == ord('='):
                    self.scale_factor += 0.1
                    print(f"Scale: {self.scale_factor:.1f}x")
                elif key == ord('-') or key == ord('_'):
                    if self.scale_factor > 0.2:
                        self.scale_factor -= 0.1
                        print(f"Scale: {self.scale_factor:.1f}x")
                elif key == ord('['):
                    self.canny_threshold1 = max(10, self.canny_threshold1 - 10)
                    print(f"Canny low threshold: {self.canny_threshold1}")
                elif key == ord(']'):
                    self.canny_threshold1 += 10
                    print(f"Canny low threshold: {self.canny_threshold1}")
                elif key == ord('{'):
                    self.canny_threshold2 = max(self.canny_threshold1 + 10, self.canny_threshold2 - 10)
                    print(f"Canny high threshold: {self.canny_threshold2}")
                elif key == ord('}'):
                    self.canny_threshold2 += 10
                    print(f"Canny high threshold: {self.canny_threshold2}")
                elif key == ord('<') or key == ord(','):
                    self.curvature_threshold = max(0.001, self.curvature_threshold - 0.005)
                    print(f"Curvature threshold: {self.curvature_threshold:.3f}")
                elif key == ord('>') or key == ord('.'):
                    self.curvature_threshold += 0.005
                    print(f"Curvature threshold: {self.curvature_threshold:.3f}")
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"camera_snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, original_frame)
                    print(f"Snapshot saved: {filename}")
                    
        finally:
            # Clean up resources
            if self.cap is not None:
                self.cap.release()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()
            print("Camera stream closed")

def list_available_cameras():
    """List all available cameras connected to the system."""
    available_cameras = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

if __name__ == "__main__":
    # Check command-line arguments
    camera_index = 0  # Default camera
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            cameras = list_available_cameras()
            if cameras:
                print(f"Available cameras: {cameras}")
            else:
                print("No cameras detected")
            sys.exit(0)
        else:
            try:
                camera_index = int(sys.argv[1])
            except ValueError:
                print(f"Invalid camera index: {sys.argv[1]}")
                print("Usage: python camera_stream.py [camera_index | --list]")
                sys.exit(1)
    
    # Create and run camera stream
    stream = CameraStream(camera_index)
    stream.run()