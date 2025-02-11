# THIS IS ANTHONY FURCAL'S CODE FOR THE MOVING CIRCLE DETECTION ASSIGNMENT


from types import NoneType

import cv2
import numpy as np

# Getting the Camera
cap = cv2.VideoCapture('VideoFootage/LaneFootage.mp4')


def stream_processing():
    while cap.isOpened():

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding the image to highlight lines (binary image)
        _, thresh = cv2.threshold(blur, 125, 135, cv2.THRESH_BINARY)
        masked_image = cv2.bitwise_and(blur, blur, mask=thresh)

        edges = cv2.Canny(masked_image, 25, 85)

        # Four corners of the trapezoid-shaped region of interest
        # You need to find these corners manually.
        roi_points = np.array([
            (294, 154),  # Top-left corner
            (200, 237),  # Bottom-left corner
            (465, 237),  # Bottom-right corner
            (381, 154)  # Top-right corner
        ])

        cv2.polylines(frame, [roi_points], True, (0, 255, 0), 5)



        # Displays the processed frame

        cv2.imshow('Photo', frame)

        # Inputting q on the keyboard ends the program

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Closing the Window
    cv2.destroyAllWindows()
    cap.release()


stream_processing()

