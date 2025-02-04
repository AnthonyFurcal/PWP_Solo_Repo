import cv2
import numpy as np

# Getting the Camera
cap = cv2.VideoCapture(0)


def stream():
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Thresholding the image to highlight lines (binary image)
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

        # Blurring the image to reduce noise that makes it more difficult to process

        blur = cv2.GaussianBlur(thresh, (5, 5), 0)

        # Identifies the edges of all the objects in the image, including drawn lines which we are looking for here.

        edges = cv2.Canny(blur, 100, 200)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        avg_contour = np.zeros((len(contours), 2), dtype=np.float32)
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                avg_contour[i] = [cx, cy]

        # Convert to a numpy array
        avg_contour = np.array(avg_contour, dtype=np.int32)

        # Draw the average contour (optional)
        cv2.drawContours(frame, [avg_contour], -1, (0, 255, 0), 2)

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
