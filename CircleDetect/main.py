# THIS IS ANTHONY FURCAL'S CODE FOR THE MOVING CIRCLE DETECTION ASSIGNMENT


from types import NoneType

import cv2
import numpy as np

# Getting the Camera
cap = cv2.VideoCapture('SodaPhotos/IMG_3529.mov')


def stream_processing():
    while cap.isOpened():

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding the image to highlight lines (binary image)
        _, thresh = cv2.threshold(blur, 125, 135, cv2.THRESH_BINARY)
        masked_image = cv2.bitwise_and(blur, blur, mask=thresh)

        edges = cv2.Canny(masked_image, 25, 85)

        # Blurring the image to reduce noise that makes it more difficult to process

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 455,
                                   param1=155, param2=115, minRadius=95, maxRadius=0)

        try:

            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 5)
                # draw the center of the circle
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 5)

        # This is an exception in case no circles are detected in the frame
        except TypeError:

            print("No Circles Found")

        # Displays the processed frame

        cv2.imshow('Photo', frame)

        # Inputting q on the keyboard ends the program

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Closing the Window
    cv2.destroyAllWindows()
    cap.release()


stream_processing()
