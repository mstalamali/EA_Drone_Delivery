from enum import Enum
from math import atan2, pi, radians
import numpy as np
from PIL import ImageTk

def norm(vector):
    return (sum(x ** 2 for x in vector)) ** 0.5


def get_orientation_from_vector(vector):
    angle = atan2(vector[1], vector[0])
    return (360 * angle / (2 * pi)) % 360


def rotation_matrix(angle):
    theta = radians(angle)
    return np.array(((np.cos(theta), -np.sin(theta)),
                     (np.sin(theta), np.cos(theta))))


def rotate(vector, angle):
    rot_mat = rotation_matrix(angle)
    rotated_vector = rot_mat.dot(vector)
    return rotated_vector


def distance_between(robot1, robot2):
    return norm(robot2.pos - robot1.pos)

class CommunicationState(Enum):
    OPEN = 1
    CLOSED = 2

def rotatedPhotoImage(img, angle):
    angleInRads = angle * pi / 180
    diagonal = sqrt(img.width()**2 + img.height()**2)
    xmidpoint = img.width()/2
    ymidpoint = img.height()/2
    newPhotoImage = PhotoImage(width=int(diagonal), height=int(diagonal))
    for x in range(img.width()):
        for y in range(img.height()):

            # convert to ordinary mathematical coordinates
            xnew = float(x)
            ynew = float(-y)

            # shift to origin
            xnew = xnew - xmidpoint
            ynew = ynew + ymidpoint

            # new rotated variables, rotated around origin (0,0) using simoultaneous assigment
            xnew, ynew = xnew*cos(angleInRads) - ynew*sin(angleInRads), xnew * sin(angleInRads) + ynew*cos(angleInRads)

            # shift back to quadrant iv (x,-y), but centered in bigger box
            xnew = xnew + diagonal/2
            ynew = ynew - diagonal/2

            # convert to -y coordinates
            xnew = xnew
            ynew = -ynew

            # get pixel data from the pixel being rotated in hex format
            rgb = '#%02x%02x%02x' % img.get(x, y)

            # put that pixel data into the new image
            newPhotoImage.put(rgb, (int(xnew), int(ynew)))

            # this helps fill in empty pixels due to rounding issues
            newPhotoImage.put(rgb, (int(xnew+1), int(ynew)))
