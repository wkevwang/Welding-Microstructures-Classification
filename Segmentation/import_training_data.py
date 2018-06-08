import os
from scipy.misc import imread
import numpy as np
import cv2

RAW_IMAGES_DIRECTORY = './Data/'
IMAGE_NAMES = []
CLASS_NAMES = [
    'PF',
    'FS',
    'AF',
    'FS-AF',
    'Other'
]

def load_data():
    images = []
    masks = []
    for image_name in IMAGE_NAMES:
        print("Loading", image_name)
        image = cv2.imread(os.path.join(RAW_IMAGES_DIRECTORY, image_name))
        try:
            PF_mask = imread(os.path.join(RAW_IMAGES_DIRECTORY, image_name.split('.')[0] + '_PF.png'))
        except FileNotFoundError:
            PF_mask = np.zeros((1920, 2560))
            print("No PF mask for", image_name)
        try:
            FS_mask = imread(os.path.join(RAW_IMAGES_DIRECTORY, image_name.split('.')[0] + '_FS.png'))
        except FileNotFoundError:
            FS_mask = np.zeros((1920, 2560))
            print("No FS mask for", image_name)
        try:
            AF_mask = imread(os.path.join(RAW_IMAGES_DIRECTORY, image_name.split('.')[0] + '_AF.png'))
        except FileNotFoundError:
            AF_mask = np.zeros((1920, 2560))
            print("No AF mask for", image_name)
        try:
            FS_AF_mask = imread(os.path.join(RAW_IMAGES_DIRECTORY, image_name.split('.')[0] + '_FS-AF.png'))
        except FileNotFoundError:
            FS_AF_mask = np.zeros((1920, 2560))
            print("No FS/AF mask for", image_name)
        mask = np.zeros((1920, 2560, 4))
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if PF_mask[r, c] == 255:
                    mask[r, c, 0] = 1
                elif FS_mask[r, c] == 255:
                    mask[r, c, 1] = 1
                elif AF_mask[r, c] == 255:
                    mask[r, c, 2] = 1
                elif FS_AF_mask[r, c] == 255:
                    mask[r, c, 2] = 1
                else:
                    mask[r, c, 3] = 1
        mask = mask.astype(np.uint8)
        images.append(image)
        masks.append(mask)
    return (images, masks, CLASS_NAMES)

def preprocess_data(images, masks):
    print("Preprocessing data...")
    RESIZE_RATIO = 2
    GRID_ROWS = 6
    GRID_COLUMNS = 8
    images_processed = []
    masks_processed = []
    for index in range(len(images)):
        image = images[index]
        mask = masks[index]
        image_resized = cv2.resize(image, (int(2560 / RESIZE_RATIO), int(1920 / RESIZE_RATIO)))
        mask_resized = cv2.resize(mask, (int(2560 / RESIZE_RATIO), int(1920 / RESIZE_RATIO)))
        for i in range(GRID_ROWS):
            for j in range(GRID_COLUMNS):
                y_min = i * image_resized.shape[0] // GRID_ROWS
                y_max = (i + 1) * image_resized.shape[0] // GRID_ROWS
                x_min = j * image_resized.shape[1] // GRID_COLUMNS
                x_max = (j + 1) * image_resized.shape[1] // GRID_COLUMNS
                image_crop = image_resized[y_min:y_max, x_min:x_max]
                mask_crop = mask_resized[y_min:y_max, x_min:x_max]
                for r in range(4):
                    M = cv2.getRotationMatrix2D((image_crop.shape[1]/2, image_crop.shape[0]/2), 90*r, 1)
                    image_rotated = cv2.warpAffine(image_crop, M, (image_crop.shape[1], image_crop.shape[0]))
                    mask_rotated = cv2.warpAffine(mask_crop, M, (mask_crop.shape[1], mask_crop.shape[0]))
                    image_rotated = image_rotated[1:image_rotated.shape[0] - 1, 1:image_rotated.shape[1] - 1] # crop 1 pixel to avoid black border due to rotations
                    mask_rotated = mask_rotated[1:mask_rotated.shape[0] - 1, 1:mask_rotated.shape[0] - 1]
                    images_processed.append(image_rotated)
                    masks_processed.append(mask_rotated)
    return (images_processed, masks_processed)