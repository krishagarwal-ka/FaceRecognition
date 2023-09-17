import dlib
import cv2
import numpy as np
import pickle
import datetime
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

file = open('database.csv', 'r')
count = (len(file.readlines()))
file.close()

database = open('database.csv', 'a')

with open('facelist', 'rb') as x:
    known_faces = pickle.load(x)

with open('namelist', 'rb') as x:
    known_face_names = pickle.load(x)

buffer = []
for i in range (len(known_faces)):
    buffer.append(0)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        face_descriptor = face_recognition_model.compute_face_descriptor(frame, landmarks)

        threshold = 0.45

        for i, known_face_descriptor in enumerate(known_faces):
            distance = np.linalg.norm(np.array(face_descriptor) - np.array(known_face_descriptor))

            if distance < threshold:
                name = known_face_names[i]

                if time.time()-buffer[i] > 60:
                    buffer[i] = time.time()
                    print (name)
                    database.write('\n'+str(count)+','+name+','+datetime.datetime.now().strftime('%x')+','+datetime.datetime.now().strftime('%X'))
                    count += 1
                    database.close()
                    database = open('database.csv', 'a')
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_image = frame[y:y + h, x:x + w]
                    filename = name+' '+datetime.datetime.now().strftime('%X')+' '+datetime.datetime.now().strftime('%H')+datetime.datetime.now().strftime('%M')+'.jpg'
                    cv2.imwrite(filename, face_image)

                cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Facial Identification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
database.close()
