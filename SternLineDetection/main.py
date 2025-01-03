# THIS IS ANTHONY FURCAL'S PROGRAMMING PROJECT FOR THE PARALLEL LINE AND FINDING CENTER LINE WINTER BREAK ASSIGNMENT


from types import NoneType

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Getting the Camera
cap = cv2.VideoCapture(0)


# This function takes information from a list of line entered and calculates an average line based on the means of each coordinate over the list
def average_lines(lines, frame_shape):
    if not lines:
        return None

    # Creates lists to store sets of each value

    x1s, y1s, x2s, y2s = [], [], [], []

    # Goes through inputted list of lines and organizes each of their coordinate values into their corresponding lists

    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)

    # Averages out each value and returns them in a list

    x1 = int(np.mean(x1s))
    y1 = int(np.mean(y1s))
    x2 = int(np.mean(x2s))
    y2 = int(np.mean(y2s))

    return [x1, y1, x2, y2]

# This Function gets, processes, and displays the video stream from the webcam
def stream():

    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Thresholding the image to highlight lines (binary image)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Blurring the image to reduce noise that makes it more difficult to process

        blur = cv2.GaussianBlur(thresh, (5, 5), 7)

        # Identifies the edges of all the objects in the image, including drawn lines which we are looking for here.

        edges = cv2.Canny(blur, 100, 200, apertureSize=3)


        #These next three lines carve a smaller portion out of the frame that will be sent through processing code to look for lines. This is to avoid information from outside the object of interest from being detected. Output is still displayed with full frame visible

        height, width = edges.shape
        mask = np.zeros_like(edges)
        mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = edges[height // 4:3 * height // 4, width // 4:3 * width // 4]

        # OpenCV's function which scans an image for lines

        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 60, minLineLength=50, maxLineGap=5000)

        # Creating variables to store line information, list of right and left lines and midpoint variables to help organize the individual line pairs into each

        left_lines = []
        right_lines = []
        x_midpoint = 0
        y_midpoint = 0

        #A try statement is used to handle the situation that no lines were detected

        try:

            #Takes each line, and searches the rest of list returned by the HoughLines function for another line with the same slope

                for i, line1 in enumerate(lines):
                    x1, y1, x2, y2 = line1[0]
                    for j, line2 in enumerate(lines):
                        if i == j:
                            continue
                        x3, y3, x4, y4 = line2[0]

                        slope1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else 0
                        slope2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else 0

                        #Once a parallel pair is found, each of the two lines are then used to calculate both the x and y midpoints which will be used to sort lines into right and left

                        if abs(slope1 - slope2) <= np.deg2rad(50):
                            x_midpoint = (x1 + x3) / 2
                            y_midpoint = (y1 + y3) / 2

                            #This conditional checks whether the x or the y value vary more in order to determine whether to use the x or y midpoint to determine the orientation of the lanes

                            if abs(x3 - x1) > abs(y3 - y1):

                                #A minimum amount of space is required for lines to be a pair in order to avoid the sides of one line being detected separately

                                if x1 > x_midpoint and abs(x3 - x1) > 25:
                                    right_lines.append(line1)
                                    left_lines.append(line2)
                                elif x1 < x_midpoint and abs(x3 - x1) > 25:
                                    right_lines.append(line2)
                                    left_lines.append(line1)
                            else:
                                if y1 > y_midpoint and abs(y3 - y1) > 25:
                                    right_lines.append(line1)
                                    left_lines.append(line2)
                                elif y1 < y_midpoint and abs(y3 - y1) > 25:
                                    right_lines.append(line2)
                                    left_lines.append(line1)


        except TypeError:
            print('no lines')

        # Average function is used to calculate the left and right overlay lines

        left_lane = average_lines(left_lines, frame.shape)
        right_lane = average_lines(right_lines, frame.shape)

        # THe centerline is calculated using the averages of the coordinates in the two other lines

        if left_lane is not None and right_lane is not None:
            centerline = [(left_lane[0] + right_lane[0]) // 2, (left_lane[1] + right_lane[1]) // 2,
                          (left_lane[2] + right_lane[2]) // 2, (left_lane[3] + right_lane[3]) // 2]
        else:
            centerline = None

        if left_lane is not None:
            cv2.line(frame, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 5)
        if right_lane is not None:
            cv2.line(frame, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (255, 0, 0), 5)
        if centerline is not None:
            cv2.line(frame, (centerline[0], centerline[1]), (centerline[2], centerline[3]), (0, 255, 0), 5)

        #The camera feed with the line overlay is shown to the user

        cv2.imshow('Webcam', frame)

        # Inputting q on the keyboard ends the program

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Closing the Window
    cap.release()
    cv2.destroyAllWindows()

#Running the program's primary function

stream()







