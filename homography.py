import cv2
import numpy as np

# --- Configuration ---
MIN_MATCH_COUNT = 10       # Minimum number of good matches to estimate homography
KEY_TO_CYCLE = ord('n')    # Press 'n' to capture reference / reset
KEY_TO_QUIT = ord('q')     # Press 'q' to quit
WINDOW_NAME = "Camera Homography - Press 'n' to capture reference/reset, 'q' to quit"
LOWE_RATIO_TEST = 0.75     # For filtering good matches (David Lowe's ratio test)

# --- Global Variables ---
current_state = 0          # 0: WAITING_FOR_REFERENCE, 1: TRACKING
reference_frame = None
reference_kp = None
reference_des = None
orb = None                 # ORB detector
bf_matcher = None          # Brute-Force matcher

def initialize_features():
    """Initialize feature detector and matcher."""
    global orb, bf_matcher
    orb = cv2.ORB_create(nfeatures=2000) # Increase nfeatures if needed
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # crossCheck=False for ratio test

def draw_text_on_image(image, text, org=(20, 30), color=(0, 255, 0), bg_color=(0,0,0), scale=0.6, thickness=1):
    """ Helper to draw text with a background for better visibility. """
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (org[0], org[1] - text_height - baseline), (org[0] + text_width, org[1] + baseline), bg_color, -1)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def main():
    global current_state, reference_frame, reference_kp, reference_des

    initialize_features()

    cap = cv2.VideoCapture(0) # 0 for default camera
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    cv2.namedWindow(WINDOW_NAME)
    print(f"--- INSTRUCTIONS ---")
    print(f"Press '{chr(KEY_TO_CYCLE)}' to capture reference frame or to reset and capture a new one.")
    print(f"Press '{chr(KEY_TO_QUIT)}' to quit.")
    print(f"--------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        frame_display = frame.copy() # Work on a copy

        if current_state == 0: # STATE_WAITING_FOR_REFERENCE
            draw_text_on_image(frame_display, "STATE: Waiting for Reference. Aim camera and press 'n'.")

        elif current_state == 1: # STATE_TRACKING
            if reference_frame is None or reference_kp is None or reference_des is None:
                print("Error: Reference data missing in tracking state. Resetting.")
                current_state = 0
                draw_text_on_image(frame_display, "Error: No reference. Press 'n'.")
                cv2.imshow(WINDOW_NAME, frame_display)
                continue

            draw_text_on_image(frame_display, "STATE: Tracking. Press 'n' to set new reference.")

            # 1. Detect keypoints and descriptors in the current frame
            current_kp, current_des = orb.detectAndCompute(frame, None)

            if current_des is not None and len(current_des) > 0:
                # 2. Match descriptors
                # Using k-NN match for ratio test
                all_matches = bf_matcher.knnMatch(reference_des, current_des, k=2)

                # 3. Apply ratio test to find good matches
                good_matches = []
                for m_arr in all_matches:
                    if len(m_arr) == 2: # Ensure we have two neighbors
                        m, n = m_arr
                        if m.distance < LOWE_RATIO_TEST * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= MIN_MATCH_COUNT:
                    # 4. Extract location of good matches
                    src_pts = np.float32([reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # 5. Find Homography
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if H is not None:
                        # 6. Draw outline of the reference object in the current frame
                        h_ref, w_ref = reference_frame.shape[:2]
                        ref_corners = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)

                        try:
                            # Check if H is a valid 3x3 matrix before transforming
                            if H.shape == (3,3):
                                transformed_corners = cv2.perspectiveTransform(ref_corners, H)
                                frame_display = cv2.polylines(frame_display, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
                                draw_text_on_image(frame_display, f"Matches: {len(good_matches)} (Homography Found)", org=(20,60), color=(0,255,0))

                                # Optional: Warp reference image onto current frame (can be slow)
                                # warped_ref = cv2.warpPerspective(reference_frame, H, (frame.shape[1], frame.shape[0]))
                                # # Create a mask for the warped image and combine
                                # mask_warped = cv2.cvtColor(warped_ref, cv2.COLOR_BGR2GRAY)
                                # _, mask_warped = cv2.threshold(mask_warped, 1, 255, cv2.THRESH_BINARY)
                                # frame_display = cv2.copyTo(warped_ref, mask_warped, frame_display)

                            else:
                                draw_text_on_image(frame_display, f"Matches: {len(good_matches)} (Homography Error)", org=(20,60), color=(0,0,255))
                        except cv2.error as e:
                            # This can happen if H is degenerate (e.g. all points collinear)
                            print(f"cv2.perspectiveTransform error: {e}")
                            draw_text_on_image(frame_display, f"Matches: {len(good_matches)} (Transform Error)", org=(20,60), color=(0,0,255))
                    else:
                        draw_text_on_image(frame_display, f"Matches: {len(good_matches)} (Homography NOT Found)", org=(20,60), color=(255,100,0))
                else:
                    draw_text_on_image(frame_display, f"Not enough matches: {len(good_matches)}/{MIN_MATCH_COUNT}", org=(20,60), color=(0,165,255))
            else:
                draw_text_on_image(frame_display, "No features detected in current frame.", org=(20,60), color=(0,0,255))

        cv2.imshow(WINDOW_NAME, frame_display)
        key = cv2.waitKey(1) & 0xFF

        if key == KEY_TO_QUIT:
            break
        elif key == KEY_TO_CYCLE:
            if current_state == 0: # Was waiting for reference, now capture it
                reference_frame = frame.copy() # Capture the current camera frame
                # Pre-compute keypoints and descriptors for the reference frame
                reference_kp, reference_des = orb.detectAndCompute(reference_frame, None)
                if reference_des is None or len(reference_kp) < MIN_MATCH_COUNT:
                    print("Not enough features detected in the reference frame. Try again with a more textured scene.")
                    reference_frame = None # Invalidate reference
                    # Stay in state 0
                else:
                    print(f"Reference frame captured with {len(reference_kp)} keypoints.")
                    current_state = 1 # Transition to tracking state
            elif current_state == 1: # Was tracking, now reset to capture new reference
                print("Resetting. Aim to capture a new reference frame.")
                reference_frame = None
                reference_kp = None
                reference_des = None
                current_state = 0 # Transition back to waiting for reference

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
