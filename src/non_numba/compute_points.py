import math
from skimage.draw import line
import numpy as np
import cv2

def compute_end_point(center, angle, size):
    cx, cy = center
    w, h = size

    angle = angle % 360

    if 0 <= angle < 90:  # [0, 90)
        theta = math.radians(angle)
        ex = cx - (h - cy) * math.tan(theta)
        ey = h - 1
    elif 90 <= angle < 180:  # [90, 180)
        theta = math.radians(180 - angle)
        ex = 0
        ey = cy - cx * math.tan(theta)
    elif 180 <= angle < 270:  # [180, 270)
        theta = math.radians(angle - 180)
        ex = cx + cy * math.tan(theta)
        ey = 0
    else:  # [270, 360)
        theta = math.radians(360 - angle)
        ex = w - 1
        ey = cy - (w - cx) * math.tan(theta)

    ex, ey = min(max(0, ex), w - 1), min(max(0, ey), h - 1)

    return int(ex), int(ey)


# Usage
center = (64, 64)  # Center of the image
angle = 91 # Degrees
size = (128, 128)

end_point = compute_end_point(center, angle, size)
rr, cc = np.array(line(center[0], center[1], end_point[0], end_point[1]))

# Display points
# print(points.T)
canvas = np.zeros(size)
canvas[rr, cc] = 1
# convert the canvas to an image and save it
cv2.imwrite('img/canvas.png', canvas * 255)
