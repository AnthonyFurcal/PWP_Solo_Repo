# THIS IS ANTHONY FURCAL'S PROGRAMMING PROJECT FOR THE PARALLEL LINE AND FINDING CENTER LINE WINTER BREAK ASSIGNMENT


from types import NoneType

import cv2
import numpy as np

img = cv2.imread('SodaPhotos/Can.jpg')
kernel = np.ones((255,255),np.uint8)


def stream_processing():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Thresholding the image to highlight lines (binary image)
        _, thresh = cv2.threshold(gray, 123, 195, cv2.THRESH_BINARY_INV)

        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Blurring the image to reduce noise that makes it more difficult to process

        blur = cv2.GaussianBlur(closed, (7, 7), 5)

        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                                  param1=100, param2=70, minRadius=50, maxRadius=0)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 20)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 20)

                # Create a named window
        cv2.namedWindow('Photo', cv2.WINDOW_NORMAL)

        # Resize the window
        cv2.resizeWindow('Photo', 800, 600)

        cv2.imshow('Photo', img)
        cv2.waitKey(0)

        # Inputting q on the keyboard ends the program

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Closing the Window
            cv2.destroyAllWindows()




stream_processing()