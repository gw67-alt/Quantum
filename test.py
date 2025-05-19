import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect affine transformation using ORB
def detect_affine_transformation(prev_frame, current_frame):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(current_frame, None)

    if kp1 is None or kp2 is None or des1 is None or des2 is None:
        return None, None, None, 0, None, []  # Initialize matches to an empty list

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_count = len(matches)

    if match_count < 10:
        return None, kp1, kp2, match_count, None, matches

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M, kp1, kp2, match_count, mask, matches

# Function to compute spacetime keypoints (3D points: x, y, frame index)
def compute_spacetime_keypoints(kp, frame_idx):
    return np.array([[pt.pt[0], pt.pt[1], frame_idx] for pt in kp])

# Function to check the Freudenthal stability
def check_freudenthal_stability(kp1, kp2, matches, t1, t2, std_threshold=2.0):
    if len(matches) == 0 or kp1 is None or kp2 is None:
        return False, 0.0

    kp1_3d = compute_spacetime_keypoints(kp1, t1)
    kp2_3d = compute_spacetime_keypoints(kp2, t2)

    src_pts = np.array([kp1_3d[m.queryIdx] for m in matches])
    dst_pts = np.array([kp2_3d[m.trainIdx] for m in matches])

    displacements = np.linalg.norm(dst_pts - src_pts, axis=1)
    std_disp = np.std(displacements)

    return std_disp < std_threshold, std_disp

# Function to process the frame, calculate displacements, and display Freudenthal stability
def process_frame(frame, prev_frame, kp1, kp2, affine_matrix, homography_matrix, mask, matches, freudenthal_status, motion_std):
    vis = frame.copy()

    if homography_matrix is not None and kp1 is not None and kp2 is not None and mask is not None and matches is not None:
        top_n = 20  # You can adjust the number of matches to display
        limited_matches = matches[:top_n]
        limited_mask = mask.ravel().tolist()[:top_n]

        draw_params = dict(
            matchColor=(0, 255, 0),  # Green matches
            singlePointColor=None,
            matchesMask=limited_mask,  # Only draw the inliers
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        )

        vis_matches = cv2.drawMatches(prev_frame, kp1, frame, kp2, limited_matches, None, **draw_params)
        vis = vis_matches

        # Display the homography matrix and affine transformation
        cv2.putText(vis, "Homography Matrix:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        matrix_str = str(homography_matrix.round(2)).replace('\n', ' ')
        cv2.putText(vis, matrix_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)

        if affine_matrix is not None:
            tx = affine_matrix[0, 2]
            ty = affine_matrix[1, 2]
            cv2.putText(vis, f"Affine Translation: ({tx:.2f}, {ty:.2f})", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Freudenthal motion stability
        status_text = "Stable" if freudenthal_status else "Unstable"
        cv2.putText(vis, f"Freudenthal Stability: (std={motion_std:.2f})", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(vis, "No Homography Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return vis
def run_video_processing():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_frame = None
    frame_idx = 0
    displacement_list = []  # Store the displacement values

    # Initialize interactive plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], label='Motion Std', color='blue')  # Initialize with empty lists
    ax.set_title("Motion Standard Deviation (Freudenthal Stability)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Standard Deviation")
    ax.legend()
    plt.show()

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        current_frame = cv2.resize(current_frame, (640, 480))
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        freudenthal_status = False
        motion_std = 0.0
        affine_matrix = None
        homography_matrix = None
        kp1, kp2, mask, matches = None, None, None, [] # Initialize matches to empty list

        if prev_frame is not None:
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            homography_matrix, kp1, kp2, match_count, mask, matches = detect_affine_transformation(gray_prev, gray_current)

            if homography_matrix is not None and np.allclose(homography_matrix[2, :], [0, 0, 1], atol=0.1):
                affine_matrix = homography_matrix[:2, :]

            freudenthal_status, motion_std = check_freudenthal_stability(kp1, kp2, matches, frame_idx - 1, frame_idx)
            displacement_list.append(motion_std)

            # Update the plot in real-time
            line.set_xdata(range(len(displacement_list)))  # X-axis is the frame number
            line.set_ydata(displacement_list)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Process frame and display
        processed_frame = process_frame(current_frame, prev_frame if prev_frame is not None else current_frame,
                                        kp1, kp2, affine_matrix, homography_matrix, mask, matches,
                                        freudenthal_status, motion_std)

        cv2.imshow("Freudenthal Homotopy Motion Analysis", processed_frame)
        prev_frame = current_frame.copy()
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video_processing()