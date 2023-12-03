import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

emotion_dict= {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
classifier = load_model('model.h5')

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
count=0
p=0
emotions=[]
i=0
class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        global count
        global emotions
        global i
        global im
        global lp
        global p
        img = frame.to_ndarray(format="bgr24")
        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            cv2.putText(img,"No face detected",(50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)          
        for (x, y, w, h) in faces:
            count+=1
            p=int((count/500)*100)
            s="SCANNING.. "+str(p)+"%"
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                emotions.append(maxindex) 
            label_position = (x, y)
            im=img
            lp=label_position
            if (p<=100):
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, s, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
            else:
                count_negative=emotions.count(0)+emotions.count(1)+emotions.count(2)+emotions.count(5)
                count_positive=emotions.count(3)+emotions.count(6)
                cv2.putText(img, "SCANNING STOPPED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)
                if(count_negative>count_positive):
                    cv2.putText(img, "STRESSED", lp, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(img, "NO STRESS", lp, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # page1.py
        return img
def main():
    # Face Analysis Application #
    activiteis = ["Home", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    if choice == "Home":
        st.title("We think you are stressed.Do you need further help?")
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        ctx=webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=Faceemotion)
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    This web app can be used as mental health analyzing applications by integrating with any social media app</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)
    else:
        pass
if __name__ == "__main__":
    main()


