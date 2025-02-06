import cv2
import numpy as np
from scipy.interpolate import CubicSpline

# Getting the Camera
cap = cv2.VideoCapture(0)

BGS = cv2.createBackgroundSubtractorMOG2()

def stream():
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Thresholding the image to highlight lines (binary image)
        _, thresh = cv2.threshold(gray, 110, 160, cv2.THRESH_BINARY)

        # Blurring the image to reduce noise that makes it more difficult to process

        blur = cv2.GaussianBlur(thresh, (5, 5), 0)


        # Identifies the edges of all the objects in the image, including drawn lines which we are looking for here.

        edges = cv2.Canny(blur, 200, 255)

        height, width = edges.shape
        mask = np.zeros_like(edges)
        mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = edges[height // 4:3 * height // 4, width // 4:3 * width // 4]

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

            # Interpolate points to increase the number of contour points for smoother centerline
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
                cv2.line(frame, tuple(centerline[i - 1]), tuple(centerline[i]), (255, 0, 0), 2)

        # Step 3: Draw the contours on the image (different colors for left and right)
        cv2.drawContours(frame, left_contours, -1, (0, 255, 0), 2)  # Green for left
        cv2.drawContours(frame, right_contours, -1, (0, 0, 255), 2)  # Red for right

        cv2.imshow('Webcam', frame)
        cv2.imshow('Webcam2', mask)

        # Inputting q on the keyboard ends the program

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Closing the Window
    cap.release()
    cv2.destroyAllWindows()


# Running the program's primary function

stream()
