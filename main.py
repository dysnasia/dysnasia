from matplotlib import image
from cv2 import VideoCapture
import face_recognition
import cv2
import numpy as np

path = 'C:\\Users\\ariel\\Desktop\\Desktop\\Coding Projects\\PISP\\Face Recognition Attendance Checker'

#Grab the webcam
cap = cv2.VideoCapture(0)

# Load a sample image / Learn how to recognize it
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample image / Learn how to recognize it
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create the array for known faces
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]

known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# Starting some variables
face_locations = []
face_encodings = []
face_names = []

while True:
    # Grabbing the single frame
    ret, frame = cap.read()

    #Resizing the image
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

    #Change the colors of the image for face_recognition use
    rgb_small_frame = small_frame[:, :, ::-1]

    #Find the faces on the webcam
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        #Test for match
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # To display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale the image for the user
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Drawing the box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # Quitting the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()