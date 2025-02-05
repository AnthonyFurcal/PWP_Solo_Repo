import cv2
import numpy as np

# Getting the Camera
cap = cv2.VideoCapture(0)


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

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

        # Step 3: Draw the contours on the image (different colors for left and right)
        cv2.drawContours(frame, left_contours, -1, (0, 255, 0), 2)  # Green for left
        cv2.drawContours(frame, right_contours, -1, (0, 0, 255), 2)  # Red for right


        cv2.imshow('Webcam', frame)
        cv2.imshow('Webcam2', edges)

        # Inputting q on the keyboard ends the program

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Closing the Window
    cap.release()
    cv2.destroyAllWindows()


# Running the program's primary function

stream()
