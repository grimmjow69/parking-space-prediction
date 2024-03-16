import cv2
import numpy as np
import json

rectangles = []
current_rectangle = []
drawing = False


def draw_rectangle(event, x, y, flags, param):
    global drawing, current_rectangle, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_rectangle = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rectangle.append((x, y))
        rectangles.append(tuple(current_rectangle))
        cv2.rectangle(img, current_rectangle[0], current_rectangle[1], (0, 255, 0), 2)

    elif drawing and event == cv2.EVENT_MOUSEMOVE:
        img_copy = img.copy()
        cv2.rectangle(img_copy, current_rectangle[0], (x, y), (0, 255, 0), 2)
        cv2.imshow('Image', img_copy)


img = cv2.imread('images/...')

cv2.imshow('Image', img)

cv2.setMouseCallback('Image', draw_rectangle)

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

rectangles_formatted = [
    [
        [rect[0][0], rect[0][1]],  # bottom-left
        [rect[1][0], rect[0][1]],  # bottom-right
        [rect[1][0], rect[1][1]],  # top-right
        [rect[0][0], rect[1][1]]  # top-left
    ]
    for rect in rectangles
]

with open('rectangles.json', 'w') as file:
    json.dump(rectangles_formatted, file)

cv2.destroyAllWindows()
