import streamlit as st
#import tensorflow.keras
import io
import tensorflow
import keras
from PIL import Image, ImageOps
import numpy as np
import time

st.title("DetAll: All HealthCare Detection Tools at one place")
st.write("------------------------------------------")
st.sidebar.title("Command Bar")
choices = ["Home","Eyes", "Skin"]
menu = st.sidebar.selectbox("Menu: ", choices)
st.set_option('deprecation.showfileUploaderEncoding', False)
if menu =="Home":
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text("Setting up the magic...")
    time.sleep(1)
    status_text.success("All Set!")
    st.write("---------------------------------")
    st.write(" Explore the sections to your left sidebar. ")

elif menu == "Eyes":
    st.sidebar.write("It analyzes cataract, diabetic retinopathy and redness levels. Upload or take an image to get started.")
    st.write("---------------------------")
    image_input = st.file_uploader("Choose an eye image: ", type=['png', 'jpg'])
    start_camera = st.checkbox("Start Camera")

    if image_input:
        
            img = image_input.getvalue()
            st.image(img, width=300)#, height=300)
            detect = st.button("Run Analysis using uploaded model")
            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('eye_models/cataract/model.h5')
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = Image.open(image_input)
            size = (224, 224)
            image = ImageOps.fit(image, size,Image.LANCZOS)  # Image.ANTIALIAS) #PIL.Image.LANCZOS
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            
            st.write("------------------------------------------------------")
          
            if detect:
                prediction = model.predict(data)
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class1 > 3*class2:
                    st.markdown("Your Model predicts the eye cloudiness risk is {:.2f}%".format(class1 * 100) )
                elif class2 > 3*class1:
                    st.markdown("Your model thinks the eyes are clear with confidence {:.2f}%".format(class2 * 100))
                else:
                    st.write("We encountered an ERROR in making a definite prediction. This should be temporary, please try again with a better quality image.")

    if start_camera:
        picture = st.camera_input("Take a picture", key="eye_photo" ,help="Click a close up photo of your eye so that we can check and analyse it")
        if picture:
            img = picture.getvalue()
            st.image(img, width=300)#, height=300)
            detect = st.button("Run Analysis using uploaded model")
            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('eye_models/cataract/model.h5')
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = Image.open(picture)
            size = (224, 224)
            image = ImageOps.fit(image, size,Image.LANCZOS)  # Image.ANTIALIAS) #PIL.Image.LANCZOS
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            
            st.write("------------------------------------------------------")
          
            if detect:
                prediction = model.predict(data)
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class1 > 3*class2:
                    st.markdown("Your Model predicts the eye cloudiness risk is {:.2f}%".format(class1 * 100) )
                elif class2 > 3*class1:
                    st.markdown("Your model does not  by {:.2f}%".format(class2 * 100))
                else:
                    st.write("We encountered an ERROR in making a definite prediction. This should be temporary, please try again with a better quality image.")
elif menu == "Skin":
    st.sidebar.write("Get started.")
    st.write("---------------------------")
    image_input = st.file_uploader("Choose an eye image: ", type=['png', 'jpg'])
    start_camera = st.checkbox("Start Camera")

    if image_input:
            img = image_input.getvalue()
            analyze = st.sidebar.button("Analyze")
           
            st.image(img)
            st.write("-----------------------------------------")
            np.set_printoptions(suppress=True)
            #model = tensorflow.models.load_model('model.h5')
            model = tensorflow.keras.models.load_model('best_model (1).h5')


            
            data = np.ndarray(shape=(1, 28, 28, 3), dtype=np.float32)
            if analyze: 
                image = Image.open(image_input)
                size = (28, 28)
                image = ImageOps.fit(image, size, Image.LANCZOS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                st.write(prediction)
                st.write("0/1/3/5: ", prediction[0,0]+prediction[0,1]+prediction[0,3]+prediction[0,5], "2: ", prediction[0,2], "4/6: ", prediction[0,4] + prediction[0,6])
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                st.markdown("*0:* Actinic Keratoses and Intraepithelial Carcinomae (akiec)")
                st.markdown("*1:* Basal Cell Carcinoma (bcc)")
                st.markdown("*2:* Benign Keratosis-like Lesions (bkl)")
                st.markdown("*3:* Dermatofibroma (df)")
                st.markdown("*4:* Melanocytic Nevi (nv)")
                st.markdown("*5:* Pyogenic Granulomas and Hemorrhage (vasc)")
                st.markdown("*6:* Melanoma (mel)")
                st.markdown(''' 
                        :red[NOTE] \n 
                         1. Please note that :blue[akiec, bcc, df and vasc] may look similar in photos, therefore we have combined their probabilities
                         2. Similarly :blue[nv and mel] may look similar in photos, therefore we have combined their probabilities''')
                if class1 - class2 > 0.5:
                    st.markdown("**Benign Detected.** Confidence: {:.2f}%".format(class1 * 100))
                elif class2 - class1 > 0.5:
                    st.markdown("**Malign Detected.** Confidence: {:.2f}".format(class2 * 100))
                else:
                    st.write("Error! Please upload a better quality image for accuracy.")

    if(start_camera):
        picture = st.camera_input("Take a picture", key="eye_photo" ,help="Click a close up photo of your skin so that we can check and analyse it")
        if picture:
            img = picture.getvalue()
            analyze = st.sidebar.button("Analyze")
           
            st.image(img)
            st.write("-----------------------------------------")
            np.set_printoptions(suppress=True)
            #model = tensorflow.models.load_model('model.h5')
            model = tensorflow.keras.models.load_model('best_model (1).h5')


            
            data = np.ndarray(shape=(1, 28, 28, 3), dtype=np.float32)
            if analyze: 
                image = Image.open(picture)
                size = (28, 28)
                image = ImageOps.fit(image, size, Image.LANCZOS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                st.write(prediction)
                st.write("0/1/3/5: ", prediction[0,0]+prediction[0,1]+prediction[0,3]+prediction[0,5], "2: ", prediction[0,2], "4/6: ", prediction[0,4] + prediction[0,6])
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                st.markdown("*0:* Actinic Keratoses and Intraepithelial Carcinomae (akiec)")
                st.markdown("*1:* Basal Cell Carcinoma (bcc)")
                st.markdown("*2:* Benign Keratosis-like Lesions (bkl)")
                st.markdown("*3:* Dermatofibroma (df)")
                st.markdown("*4:* Melanocytic Nevi (nv)")
                st.markdown("*5:* Pyogenic Granulomas and Hemorrhage (vasc)")
                st.markdown("*6:* Melanoma (mel)")
                st.markdown(''' 
                        :red[NOTE] \n 
                         1. Please note that :blue[akiec, bcc, df and vasc] may look similar in photos, therefore we have combined their probabilities
                         2. Similarly :blue[nv and mel] may look similar in photos, therefore we have combined their probabilities''')
                if class1 - class2 > 0.5:
                    st.markdown("**Benign Detected.** Confidence: {:.2f}%".format(class1 * 100))
                elif class2 - class1 > 0.5:
                    st.markdown("**Malign Detected.** Confidence: {:.2f}".format(class2 * 100))
                else:
                    st.write("Error! Please upload a better quality image for accuracy.")
