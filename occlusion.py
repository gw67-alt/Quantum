import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading

# Maximum number of data points to store for the chart
MAX_DATA_POINTS = 100

# Global variables for data sharing between threads
occlusion_data = deque(maxlen=MAX_DATA_POINTS)
timestamps = deque(maxlen=MAX_DATA_POINTS)
running = True
start_time = time.time()

# Define figure and axes as global variables
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'b-', lw=2)

# Window size for moving average of occlusion data
WINDOW_SIZE = 50
averaged_occlusion_data = deque(maxlen=MAX_DATA_POINTS)

def detect_sphere(frame):
    """Detect spherical objects in the frame."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Apply Hough Circle Transform to detect circular objects
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=1000,
        param2=3000,
        minRadius=2,
        maxRadius=450
    )

    if circles is not None:
        # Convert to integer coordinates
        circles = np.uint16(np.around(circles))

        # Get the largest circle (assumed to be the sphere)
        largest_circle = circles[0, 0]
        center = (largest_circle[0], largest_circle[1])
        radius = largest_circle[2]

        return center, radius

    # If no circle is found, return default values
    height, width = frame.shape[:2]
    return (width // 2, height // 2), 100

def detect_lattice_points(frame):
    """Detect bright points in the frame that could represent a photonic lattice."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to identify bright spots
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to isolate individual points
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours of bright regions
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract centers of small contours (likely to be lattice points)
    lattice_points = []
    for contour in contours:
        # Calculate contour area to filter out large objects
        area = cv2.contourArea(contour)
        if area < 100:  # Only consider small bright spots
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lattice_points.append((cx, cy))

    return lattice_points

def calculate_occlusion(frame, sphere_center, sphere_radius, lattice_points):
    """Calculate the occlusion of the sphere by lattice points."""
    # Create masks
    h, w = frame.shape[:2]
    sphere_mask = np.zeros((h, w), dtype=np.uint8)
    lattice_mask = np.zeros((h, w), dtype=np.uint8)

    # Draw sphere on mask
    cv2.circle(sphere_mask, sphere_center, sphere_radius, 255, -1)

    # Draw lattice points on mask
    for point in lattice_points:
        cv2.circle(lattice_mask, point, 3, 255, -1)

    # Calculate intersection (occlusion)
    occlusion = cv2.bitwise_and(sphere_mask, lattice_mask)

    # Calculate percentage
    sphere_area = np.sum(sphere_mask > 0)
    if sphere_area == 0:
        return 0, occlusion, sphere_area

    occlusion_area = np.sum(occlusion > 0)

    # Calculate raw percentage then amplify it to make small changes more visible
    raw_occlusion_percentage = (occlusion_area / sphere_area) * 100

    # Amplify the occlusion value to make it more sensitive
    # Using a formula that makes small values more visible while still capping at 100%
    # This gives more resolution to the typically small occlusion values
    amplification_factor = 75.0  # Adjust this value to control sensitivity
    occlusion_percentage = min(raw_occlusion_percentage * amplification_factor, 100.0)

    return occlusion_percentage, occlusion, sphere_area

def run_video_processing():
    """Run real-time occlusion detection on webcam feed."""
    global occlusion_data, timestamps, running, start_time
    global averaged_occlusion_data

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        running = False
        return

    # Variables for FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    print("Running real-time occlusion detection. Press 'q' to quit.")

    # Variables for averaging/stabilizing detection
    prev_sphere_center = None
    prev_sphere_radius = None
    smoothing_factor_sphere = 0.3  # Higher value = more smoothing for sphere

    while running:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            running = False
            break

        # Create a copy for visualization
        vis_img = frame.copy()

        # Detect sphere
        current_sphere_center, current_sphere_radius = detect_sphere(frame)

        # Apply smoothing to sphere detection
        if prev_sphere_center is not None:
            sphere_center = (
                int(prev_sphere_center[0] * smoothing_factor_sphere + current_sphere_center[0] * (1 - smoothing_factor_sphere)),
                int(prev_sphere_center[1] * smoothing_factor_sphere + current_sphere_center[1] * (1 - smoothing_factor_sphere))
            )
            sphere_radius = int(prev_sphere_radius * smoothing_factor_sphere + current_sphere_radius * (1 - smoothing_factor_sphere))
        else:
            sphere_center, sphere_radius = current_sphere_center, current_sphere_radius

        prev_sphere_center = sphere_center
        prev_sphere_radius = sphere_radius

        # Detect lattice points
        lattice_points = detect_lattice_points(frame)

        # Calculate occlusion
        occlusion_pct, occlusion, sphere_area = calculate_occlusion(frame, sphere_center, sphere_radius, lattice_points)

        # Add data to the queues for plotting
        current_time = time.time() - start_time
        occlusion_data.append(occlusion_pct)
        timestamps.append(current_time)

        # Apply moving average to occlusion data
        if len(occlusion_data) >= WINDOW_SIZE:
            window = np.array(list(occlusion_data)[-WINDOW_SIZE:])
            averaged_value = np.mean(window)
            averaged_occlusion_data.append(averaged_value)
        elif occlusion_data:
            averaged_occlusion_data.append(occlusion_data[-1]) # Use last value if window not full

        # Visualize results
        # Draw sphere contour in blue
        cv2.circle(vis_img, sphere_center, int(sphere_radius), (255, 0, 0), 2)

        # Draw lattice points in green
        for point in lattice_points:
            cv2.circle(vis_img, point, 3, (0, 255, 0), -1)

        # Draw occlusion points in red
        occlusion_points = np.where(occlusion > 0)
        for y, x in zip(occlusion_points[0], occlusion_points[1]):
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                vis_img[y, x] = (0, 0, 255)

        # Calculate raw occlusion percentage for display
        if sphere_area > 0:
            occlusion_area = np.sum(occlusion > 0)
            raw_occlusion_pct = (occlusion_area / sphere_area) * 100
        else:
            raw_occlusion_pct = 0

        # Add text with both raw and amplified occlusion percentages
        cv2.putText(vis_img, f"Raw Occlusion: {raw_occlusion_pct:.2f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, f"Amplified Occlusion: {occlusion_pct:.2f}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if averaged_occlusion_data:
            cv2.putText(vis_img, f"Avg. Occlusion: {averaged_occlusion_data[-1]:.2f}%",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Calculate FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        # Display FPS
        cv2.putText(vis_img, f"FPS: {fps:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add instructions
        cv2.putText(vis_img, "Press 'q' to quit", (10, vis_img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the result
        cv2.imshow("Sphere Contour Occlusion", vis_img)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def init_chart():
    """Initialize the matplotlib chart."""
    ax.set_xlim(0, 30)  # Initial x-axis range: 30 seconds
    ax.set_ylim(0, 100)  # y-axis range: 0-100%
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplified Occlusion (%)')
    ax.set_title('Real-time Sphere Occlusion (Amplified with Moving Average)')
    ax.grid(True)

    # Add a horizontal line at original amplification factor to indicate scale
    ax.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='20% Reference')
    ax.legend(loc='upper right')

    line.set_data([], [])
    return line,

def update_chart(frame):
    """Update the chart with new data."""
    global averaged_occlusion_data, timestamps

    # Update the line data with the averaged occlusion values
    if timestamps and averaged_occlusion_data:
        x_data = list(timestamps)[:len(averaged_occlusion_data)]
        y_data = list(averaged_occlusion_data)

        # Adjust x-axis limit to show the most recent 30 seconds
        if x_data:
            max_time = max(x_data)
            ax.set_xlim(max(0, max_time - 30), max(30, max_time))

        line.set_data(x_data, y_data)

    return line,

def main():
    """Main function to start the application."""
    global running

    try:
        # Start video processing in a separate thread
        video_thread = threading.Thread(target=run_video_processing)
        video_thread.daemon = True
        video_thread.start()

        # Create animation
        ani = FuncAnimation(fig, update_chart, init_func=init_chart, interval=100, save_count=MAX_DATA_POINTS)
        plt.show()

    except KeyboardInterrupt:
        print("Exiting...")
        running = False
    finally:
        running = False
        if 'video_thread' in locals() and video_thread.is_alive():
            video_thread.join()

if __name__ == "__main__":
    main()
