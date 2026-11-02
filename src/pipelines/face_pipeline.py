import numpy as np
import streamlit as st
import dlib
import face_recognition_models
from sklearn.svm import SVC


from src.database.db import get_all_students

@st.cache_resource
def load_dlib_models():

    #it detects only faces in the image and returns bounding boxes containing the detetcted faces
    detector=dlib.get_frontal_face_detector() 

    #it identifies important point(landmarks) on a detected face
    sp=dlib.shape_predictor(
        #This tells Python where the pre-trained facial landmark model file is located(location here means path of the file in computer).
        face_recognition_models.pose_predictor_model_location()
    )

    #it converts facial features into numerical embeddings(128-dimensional face embeddings)
    facerec=dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )
    return detector, sp, facerec

#image_np is a numpy array containing the image's pixel data
def get_face_embeddings(image_np):
    detector, sp, facerec=load_dlib_models()
    faces=detector(image_np,1)#detetcts faces in the provided image

    encodings=[]#list to store all embeddings

    for face in faces:
        shape=sp(image_np,face)
        face_descriptor=facerec.compute_face_descriptor(image_np,shape,1)# it generates a 128-d numerical representation of the detected face.

        encodings.append(np.array(face_descriptor))#convert embedding into numpy array

    return encodings


#function which fetches all the students from db and trains a model on them to identify whether a person is in the image or not
@st.cache_resource
def get_trained_model():
    X=[]
    y=[]

    student_db=get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding=student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))

    if len(X)==0:
        return 0

    clf=SVC(kernel='linear', probability=True, class_weight='balanced')
    try:
        clf.fit(X,y)
    except ValueError:
        pass

    return {'clf':clf, 'X':X,'y':y}

def train_classifier():
    st.cache_resource.clear()
    model_data=get_trained_model()
    return bool(model_data)

def predict_attendance(class_image_np):
    encodings=get_face_embeddings(class_image_np)
    detected_student={}

    model_data=get_trained_model()
    if not model_data:
        return detected_student, [], len(encodings) # 1.detected student embeddings, 2.student ids, 3. number of ecodings(length of encodings)

    clf=model_data['clf']
    X_train=model_data['X']
    y_train=model_data['y']

    all_students=sorted(list(set(y_train)))

    for encoding in encodings:
        if len(all_students)>=2:
            predicted_id=int(clf.predict([encoding])[0])

        else:
            predicted_id=int(all_students[0])#that one student's id

        student_embedding=X_train[y_train.index(predicted_id)] #record index in X table and y table are same for any student

        best_match_score=np.linalg.norm(student_embedding-encoding)
        resemblance_threshold=0.6

        if best_match_score<=resemblance_threshold:
            detected_student[predicted_id]=True

    return detected_student, all_students, len(encodings)