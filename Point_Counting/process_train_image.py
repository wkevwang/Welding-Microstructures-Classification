import pyrebase
import cv2
import os
import numpy as np
import re
import sys

RADIUS = 25

image_paths = []

firebase_config = {}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
user = auth.sign_in_with_email_and_password("user@example.ca", "Password")
token = user['idToken']
db = firebase.database()

def process_labelled_patch(img, d, counts):
    print(counts)
    if d == None:
        return
    label = d['label']
    y_min = d['y'] - RADIUS
    y_max = d['y'] + RADIUS
    x_min = d['x'] - RADIUS
    x_max = d['x'] + RADIUS
    if y_min < 0 or y_max > 1920:
        return
    if x_min < 0 or x_max > 2560:
        return
    subimage = img[y_min:y_max, x_min:x_max]
    label_abbv = label.split(' - ')[0]
    if label == 'Undefined' or label_abbv == "U": 
        return
    if label_abbv == "P":
        for r in range(2):
            if label_abbv in counts:
                counts[label_abbv] += 1
            else:
                counts[label_abbv] = 1
            M = cv2.getRotationMatrix2D((subimage.shape[1]/2, subimage.shape[0]/2), 90*r, 1)
            image_rotated = cv2.warpAffine(subimage, M, (subimage.shape[1], subimage.shape[0]))
            cv2.imwrite("data/" + label_abbv + "/" + label_abbv + "." + str(counts[label_abbv]) + ".jpg", image_rotated)
    else:
        if label_abbv in counts:
            counts[label_abbv] += 1
        else:
            counts[label_abbv] = 1
        cv2.imwrite("data/" + label_abbv + "/" + label_abbv + "." + str(counts[label_abbv]) + ".jpg", subimage)

counts = {}
for image_path in image_paths:
    FILE_NAME = os.path.basename(image_path)
    IMAGE_NAME = re.sub('[^a-zA-Z0-9]', '', FILE_NAME)
    print("Processing", IMAGE_NAME)

    # Get the labels for an image
    ref = db.child('annotations').child(IMAGE_NAME)
    data = ref.get(token).val()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if not isinstance(data, (list,)): # when data is sparse, it becomes an OrderedDict
        for (key, d) in data.items(): 
            process_labelled_patch(img, d, counts)
    else:
        for d in data:
            process_labelled_patch(img, d, counts)