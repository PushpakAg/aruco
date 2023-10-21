import cv2
import cv2.aruco as aruco

# Read the input image
inputImage = cv2.imread(r'C:\Users\pushp\Documents\pythonAI\Task_2A_files\public_test_cases\aruco_1.png')

# Define the ArUco dictionary and detector parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
detector_params = aruco.DetectorParameters()

# Detect ArUco markers in the image
marker_corners, marker_ids, _ = aruco.detectMarkers(inputImage, aruco_dict, parameters=detector_params)

# Draw detected markers on the image
if marker_ids is not None:
    aruco.drawDetectedMarkers(inputImage, marker_corners, marker_ids)

    # Iterate through detected markers
    for i in range(len(marker_ids)):
        marker_id = marker_ids[i][0]
        marker_corner_points = marker_corners[i][0]

        print(f"Marker ID: {marker_id}")
        for corner in marker_corner_points:
            x, y = corner
            print(f"Corner Coordinates: ({x}, {y})")

# Display the image with detected markers
cv2.imshow('ArUco Marker Detection', inputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
