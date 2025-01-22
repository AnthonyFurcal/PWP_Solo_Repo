# THIS IS ANTHONY FURCAL'S PROGRAMMING PROJECT FOR THE CIRCLE DETECTION ASSIGNMENT


from types import NoneType

import cv2
import numpy as np

img = cv2.imread('SodaPhotos/Can2.jpg')
kernel = np.ones((255, 255), np.uint8)


def stream_processing():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding the image to highlight lines (binary image)
    _, thresh = cv2.threshold(blur, 145, 255, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(blur, blur, mask=thresh)

    # Blurring the image to reduce noise that makes it more difficult to process

    circles = cv2.HoughCircles(masked_image, cv2.HOUGH_GRADIENT, 1, 255,
                               param1=100, param2=70, minRadius=50, maxRadius=0)

    try:

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 10)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 10)

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

    except TypeError:

        cv2.namedWindow('Photo', cv2.WINDOW_NORMAL)

        # Resize the window
        cv2.resizeWindow('Photo', 800, 600)

        cv2.imshow('Photo', thresh)
        cv2.waitKey(0)

        # Inputting q on the keyboard ends the program

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Closing the Window
            cv2.destroyAllWindows()
        print("No Circles Found")


stream_processing()
