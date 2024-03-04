import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import time

def main():
    st.title("Eye_Risk_Demo")
    st.write("------------------------------------------")
    st.sidebar.title("Command Bar")
    choices = ["Home","Eye-risk", "Skin"]
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
        st.write(" Explore the sections to your left sidebar.")
    elif menu == "Eye-risk":
        st.sidebar.write("Your uploaded model should have two classes: Risky/Cloudy and Low-Risk/Clear. This interface will only output for confidence levels above 75%")
        st.write("---------------------------")
        image_input = st.sidebar.file_uploader("Choose an eye image: ", type="jpg")
        if image_input:
            img = image_input.getvalue()
            st.sidebar.image(img, width=300)#, height=300)
            detect = st.sidebar.button("Run Analysis using uploaded model")
            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('eye_models/cataract/model.h5')
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = Image.open(image_input)
            size = (224, 224)
            image = ImageOps.fit(image, size,Image.LANCZOS)  # Image.ANTIALIAS) #PIL.Image.LANCZOS
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            size = st.slider("Adjust Image Size: ", 300, 1000)
            st.image(img, width=size)#, height=size)
            st.write("------------------------------------------------------")
          
            if detect:
                prediction = model.predict(data)
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class1 > 3*class2:
                    st.markdown("Your Model predicts the eye cloudiness risk is {:.2f}%".format(class1 * 100) )
                elif class2 > 3*class1:
                    st.markdown("Your model does not detect cloudiness  by {:.2f}%".format(class2 * 100))
                else:
                    st.write("We encountered an ERROR in making a definite prediction. This should be temporary, please try again with a better quality image.")
           

    
                    
    elif menu == "Skin":
        import zipfile
        with zipfile.ZipFile('model.zip', 'r') as zip_ref:
            zip_ref.extractall('')
            st.write('extracted!')
        st.sidebar.write("Get Started.")
        st.write("---------------------------")
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_input = st.sidebar.file_uploader("Choose a file: ", type='jpg')
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
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class1 - class2 > 0.5:
                    st.markdown("**Benign Detected.** Confidence: {:.2f}%".format(class1 * 100))
                elif class2 - class1 > 0.5:
                    st.markdown("**Malign Detected.** Confidence: {:.2f}".format(class2 * 100))
                else:
                    st.write("Error! Please upload a better quality image for accuracy.")


if __name__ == '__main__':
    main()
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
