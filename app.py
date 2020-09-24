import streamlit as st
import cv2
from PIL import Image
import random
import numpy as np
import base64
from io import BytesIO


from keras.models import load_model


colorr = """ <style>
body{
    background-color : #00ffff;
}
</style>
"""
st.markdown(colorr, unsafe_allow_html=True)


st.write("""
#### streamlit
## FaceRecognition App.
""")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('model_vgg16.h5')

#face Recognition with webcam
video_capture = cv2.VideoCapture(0)

def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    #crop all the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w, y+h),(0, 255,255),2)
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face
if st.button('Start'):
    cam = st.empty()
    while True:

        _, frame = video_capture.read()
        face = face_extractor(frame)
        if type(face) is np.ndarray:

            face = cv2.resize(face, (100, 100))
            im = Image.fromarray(face, 'RGB')
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
        
        
            if(pred[0][1] > pred[0][0]):
                cv2.putText(frame, 'Ashutosh Kumar',(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, 'Shivansh',(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
            #cv2.imshow('Video',frame)
            cam.image(frame, channels='BGR')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video_capture.release()
    cv2.destroyAllWindows()