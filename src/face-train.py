import cv2 as cv
import numpy as np
import os
from PIL import Image
import pickle
face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create() # Instantiate face recognizer


# Walking through directory looking for PNG/JPEG file to a list to use for training
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Current directory
folders = BASE_DIR.split("\\")
folders.pop(-1) # Go up one directory level
BASE_DIR = "\\".join(folders)
image_dir = os.path.join(BASE_DIR, "img")  # Navigate to image directory

current_id = 0 # ID for every label
label_ids = {} # dictionary with key=label and value=face. Used to train model.
y_labels = [] # labels are the names of scrub cast members
x_train = [] # store each face found in the images

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)

            # name of the character (ie. zach_braff)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            # If this character has not been seen before, give him/her an ID
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L") # convert colored image to grayscale
            #size = (550, 550)
            #final_image = pil_image.resize(size, Image.ANTIALIAS)  #resize images to have same dimensions
            image_array = np.array(pil_image, "uint8") # convert to numpy array. change to final_image if resizing

            # Collection of faces which are detected in the image (should be 1)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w] # region of interest = box around face
                x_train.append(roi)
                y_labels.append(id_)

# Save dictionary as byte stream
with open("label.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# Train the face recognizer
recognizer.train(x_train, np.array(y_labels))

# Save model
recognizer.save("trainer.yml")