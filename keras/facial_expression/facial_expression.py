#! /usr/bin/python3

import cv2
import numpy as np
from keras.models import load_model

# text and font for results
target = ['angry','disgust','fear','happy','sad','surprise','neutral']
font = cv2.FONT_HERSHEY_SIMPLEX

# load face cascade classifier (for face detection)
faceCascade = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')

# load keras model (for face classification)
model = load_model('./models/model_5-49-0.62.hdf5')
# print(model.summary())

# start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # Convert from BGR color to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect all faces in frame
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1)

    # Process all detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2,5)

        # Crop the face
        face_crop = frame[y:y+h,x:x+w]
        # Resize it to be 48x48
        face_crop = cv2.resize(face_crop,(48,48))
        # Change from BGR Color to Gray
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        # Change type to float32
        face_crop = face_crop.astype('float32')/255
        # Change to np.array
        face_crop = np.asarray(face_crop)
        # Reshape array
        face_crop = face_crop.reshape(1, 1,face_crop.shape[0],face_crop.shape[1])
        # Predict using keras model
        prediction = model.predict(face_crop)
        # pick the max as the result
        result = target[np.argmax(prediction)]
        # Show result text on frame
        cv2.putText(frame,result,(x,y), font, 1, (200,0,0), 3, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit if q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
