import cv2
import numpy as np

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Initialize the ArUco detector parameters
aruco_params = cv2.aruco.DetectorParameters_create()

# Load an image or initialize a video capture device
# Replace 'your_image_or_video_path' with the path to your image or video
cap = cv2.VideoCapture('your_image_or_video_path')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        # Draw the detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Iterate over detected markers and print their coordinates
        for i in range(len(ids)):
            marker_id = ids[i][0]
            marker_corners = corners[i][0]
            # Calculate the center of the marker
            marker_center = np.mean(marker_corners, axis=0)
            x, y = marker_center[0], marker_center[1]
            print(f"Marker ID {marker_id}: X={x}, Y={y}")

    # Display the frame
    cv2.imshow('ArUco Marker Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()
