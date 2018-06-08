import cv2
import os
import numpy as np
import re
import sys

def visualize(model, IMAGE_PATH, PATCH_SIZE):
    RADIUS = PATCH_SIZE // 2
    X_INITIAL = RADIUS
    Y_INITIAL = RADIUS
    # Label mapping
    label_mapping = {}
    img = cv2.imread(IMAGE_PATH)
    NUM_GRID_COLUMNS = img.shape[1] // 50
    NUM_GRID_ROWS = img.shape[0] // 50
    for i in range(NUM_GRID_COLUMNS):
        for j in range(NUM_GRID_ROWS):
            center_x = X_INITIAL + (i * 50)
            center_y = Y_INITIAL + (j * 50)
            y_min = center_y - RADIUS
            y_max = center_y + RADIUS
            x_min = center_x - RADIUS
            x_max = center_x + RADIUS
            if y_min < 0 or y_max > 1920:
                continue
            if x_min < 0 or x_max > 2560:
                continue
            subimage = img[y_min:y_max, x_min:x_max]
            subimage = subimage / 255
            prediction = model.predict(np.array([subimage]), batch_size=1)
            prediction = prediction[0]
            if np.argmax(prediction) == 0:
                cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), thickness=2)
            else:
                cv2.circle(img, (center_x, center_y), 3, (255, 255, 255), thickness=2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = label_mapping[np.argmax(prediction)]
            cv2.putText(img, text, (center_x + 5, center_y - 5), font, 0.3, (255,255,255), 1, cv2.LINE_AA)
    return img