import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import altair as alt
import pandas as pd
model = tf.keras.models.load_model('handwritten_digits.model')
image = Image.open('logo.jpg')
cola, colb, colc = st.columns([3,6,1])
with cola:
    st.write("")

with colb:
    st.image(image, width = 300)

with colc:
    st.write("")
menu = ["Home","About"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Home":
    image_number = st.sidebar.selectbox('Select the handwriten picture, there are 19 handwriten images that you can choose from and see how the model performs:',(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19))
    st.subheader('Selected image and its predicted numer')
    try:
    
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]

        img = np.invert(np.array([img]))
        
        prediction = model.predict(img)
        colb, colc = st.columns([6,5])
        
        with colb:
            st.write("The selected image:")
            st.image('digits/digit{}.png'.format(image_number), width = 300)
        with colc:
            st.write("The predicted number:")
            st.write(".. \n .. \n .. \n")
            
            st.write(".......................  {}  .......................".format(np.argmax(prediction)))
        
    except:
        print("Error reading image! Proceeding with next image...")
else:
    st.subheader("About")
    st.write("With a hybrid profile of data science and computer science, I’m pursuing a career in AI-driven firms. I believe in dedication, discipline, and creativity towards my job, which will be helpful in meeting your firm's requirements as well as my personal development.")
    st.write("Check out this project's [Github](https://github.com/bashirsadat/rds_nn_digits)")
    st.write(" My [Linkedin](https://www.linkedin.com/in/saadaat/)")
    st.write("See my other projects [LinkTree](https://linktr.ee/saadaat)")


