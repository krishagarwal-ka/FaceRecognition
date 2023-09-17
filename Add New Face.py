import dlib
import cv2
import pickle

with open('facelist', 'rb') as x:
    facelist = pickle.load(x)

with open('namelist', 'rb') as x:
    namelist = pickle.load(x)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

cap = cv2.VideoCapture(0)

known_face_dict = None

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        face_descriptor = face_recognition_model.compute_face_descriptor(frame, landmarks)

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        name = input("Enter your name: ").strip()

        if name:
            known_face_dict = {name: face_descriptor}

            face = list(known_face_dict.values())
            facelist.extend (face)

            name = list(known_face_dict.keys())
            namelist.extend (name)

        break

    cv2.imshow('Add Known Face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if known_face_dict is not None:
    with open('facelist', 'wb') as x:
        pickle.dump(facelist, x)

    with open('namelist', 'wb') as x:
        pickle.dump(namelist, x)

    print ('Saved')

