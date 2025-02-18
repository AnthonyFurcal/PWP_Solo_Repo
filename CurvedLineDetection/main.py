# THIS IS ANTHONY FURCAL'S CODE FOR THE MOVING CIRCLE DETECTION ASSIGNMENT


import cv2
import numpy as np
from scipy.interpolate import CubicSpline

# Getting the Camera
cap = cv2.VideoCapture('VideoFootage/car.mov')

def warp_frame(image):
    # Four corners of the trapezoid-shaped region of interest
    # You need to find these corners manually.
    roi_points = np.array([
        (312, 174),  # Top-left corner
        (192, 237),  # Bottom-left corner
        (480, 237),  # Bottom-right corner
        (380, 174)  # Top-right corner
    ])

    dest_points = np.array([
        (0, 0),
        (0, 360),
        (640, 360),
        (640, 0)
    ])

    pts = np.float32(roi_points)
    pts2 = np.float32(dest_points)

    perspesctiveMatrix = cv2.getPerspectiveTransform(pts, pts2)

    result = cv2.warpPerspective(image, perspesctiveMatrix, (640, 360))

    return result

def interpolate_contour(contour, num_points=20):
    contour_points = contour.reshape(-1, 2)
    x = contour_points[:, 0]
    y = contour_points[:, 1]

    # Fit cubic spline to the contour points
    cs_x = CubicSpline(np.arange(len(x)), x)
    cs_y = CubicSpline(np.arange(len(y)), y)

    # Generate smooth points along the spline (interpolating 500 points)
    smooth_x = cs_x(np.linspace(0, len(x) - 1, num_points))
    smooth_y = cs_y(np.linspace(0, len(y) - 1, num_points))

    return np.vstack((smooth_x, smooth_y)).astype(np.int32).T


def find_lines(filtered_image, original_image):

    contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Step 1: Compute centroids of all contours and calculate the average center point
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])  # Centroid x-coordinate
            cy = int(M["m01"] / M["m00"])  # Centroid y-coordinate
            centroids.append((cx, cy))

    # Calculate the average centroid (this will act as the center point for sorting)
    if centroids:
        avg_cx = int(np.mean([c[0] for c in centroids]))  # Average x-coordinate of centroids
        avg_cy = int(np.mean([c[1] for c in centroids]))  # Average y-coordinate of centroids
    else:
        avg_cx, avg_cy = 0, 0  # Fallback in case no contours found

    # Step 2: Sort contours based on their position relative to the average center point
    left_contours = []
    right_contours = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # Centroid x-coordinate
            cy = int(M["m01"] / M["m00"])  # Centroid y-coordinate

            # Sort contours based on whether their centroid is to the left or right of the average center point
            if cx < avg_cx:
                left_contours.append(contour)  # Contour is on the left side of the center
            else:
                right_contours.append(contour)  # Contour is on the right side of the center

            # Step 8: Generate centerline by averaging points from left and right contours
        centerline = []

        # Interpolate left and right contours
        left_contours_interp = [interpolate_contour(contour) for contour in left_contours]
        right_contours_interp = [interpolate_contour(contour) for contour in right_contours]

        # Generate the centerline by averaging corresponding points from left and right interpolated contours
        for left_contour, right_contour in zip(left_contours_interp, right_contours_interp):
            for i in range(len(left_contour)):
                left_point = left_contour[i]
                right_point = right_contour[i]

                # Calculate the midpoint between the corresponding points on the left and right contours
                midpoint = ((left_point[0] + right_point[0]) // 2, (left_point[1] + right_point[1]) // 2)
                centerline.append(midpoint)

        # Step 9: Apply Gaussian smoothing to the centerline for additional smoothness
        centerline = np.array(centerline)

        # Step 10: Draw the smoothed centerline on the image
        for i in range(1, len(centerline)):
            cv2.line(original_image, tuple(centerline[i - 1]), tuple(centerline[i]), (255, 0, 0), 2)

    # Step 3: Draw the contours on the image (different colors for left and right)
    cv2.drawContours(original_image, left_contours, -1, (0, 255, 0), 2)  # Green for left
    cv2.drawContours(original_image, right_contours, -1, (0, 0, 255), 2)  # Red for right

    return original_image


def skeletonize(image):
    skeleton = np.zeros(image.shape, np.uint8)

    eroded = np.copy(image)

    kernel = np.ones((3, 3), np.uint8)

    while True:
        opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(eroded, opening)
        eroded = cv2.erode(eroded, kernel)
        skeleton = cv2.bitwise_or(skeleton, temp)
        if np.sum(eroded) == 0:
            break

    return skeleton

def stream_processing():
    while cap.isOpened():

        ret, frame = cap.read()

        resized_frame = cv2.resize(frame, (640, 360))

        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Thresholding the image to highlight lines (binary image)
        _, mask = cv2.threshold(blur, 165, 185, cv2.THRESH_BINARY)
        masked_image = cv2.bitwise_and(blur, blur, mask=mask)

        edges = cv2.Canny(masked_image, 45, 65)

        # skeleton = skeletonize(edges)

        warped_edge = warp_frame(edges)
        warped_reg = warp_frame(resized_frame)

        with_lines = find_lines(warped_edge, warped_reg)


        # Displays the processed frame

        cv2.imshow('Photo', edges)
        cv2.imshow('Photo2', with_lines)
        print(cv2.getWindowImageRect('Photo2'))

        # Inputting q on the keyboard ends the program

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Closing the Window
    cv2.destroyAllWindows()
    cap.release()


stream_processing()

