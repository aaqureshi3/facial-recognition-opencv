import cv2 as cv
import numpy as numpy
import pickle

face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv.CascadeClassifier('cascades/data/haarcascade_eye.xml')
#smile_cascade = cv.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv.face.LBPHFaceRecognizer_create() # Instantiate face recognizer
recognizer.read("trainer.yml") # load model into recognizer



def load_labels(labels):
    new_labels = {}
    with open(labels, "rb") as f:
        og_labels = pickle.load(f)
        new_labels = {v:k for k,v in og_labels.items()}
    return new_labels


def annotate_image(image, labels):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # region of interest for gray image
        roi_gray = gray[y:y+h, x:x+w]

        # region of interest for the original colored image
        roi_color = image[y:y+h, x:x+w]

        # Recognize actor/actress by ID
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 75:
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]  # get actor/actress name using ID generated from recognizer
            color = (255,255,255)
            stroke = 2
            cv.putText(image, name, (x,y), font, 1, color, stroke, cv.LINE_AA)

        color = (255, 0, 0) # line color BGR
        stroke = 2 # line thickness
        width = x + w
        height = y + h
        cv.rectangle(image, (x,y), (width,height), color, stroke) # draw rectangle on frame

        # Find and draw eyes (just for fun)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
    return image


def display_image(image_path, labels_path):
    # Load image
    image = cv.imread(image_path)

    # Load model
    labels = load_labels(labels_path)

    # Identify faces based on model
    annotated_image = annotate_image(image, labels)

    # Display the resulting frame
    cv.imshow('image', annotated_image)
    cv.waitKey(0)


def display_webcam(labels_path):
    # Load model
    labels = load_labels(labels_path)

    cap = cv.VideoCapture(0)

    while(True):   
        # Capture frame-by-frame
        _, frame = cap.read()

        # Annotate frame
        annotated_frame = annotate_image(frame, labels)

        # Display the resulting frame
        cv.imshow('frame', annotated_frame)

        # Stop video if q is pressed
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    # When everythig is done, release teh capture
    cap.release()
    cv.destroyAllWindows()



#display_image("..\scrub1.png", "label.pickle")
#display_webcam("label.pickle")