import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
model = tf.keras.models.load_model('handwritten_digits.model')
def image_gr_28(img):
    img = img.resize((28,28), Image.ANTIALIAS).convert('L')
    return img
def load_image(image_file):
	img = Image.open(image_file)
    
	return img
image_file = st.sidebar.file_uploader("Upload your input, png, jpg or jpeg file",  type=["png","jpg","jpeg"])
if image_file is not None:
    # To View Uploaded Image
    st.write("The uploaded image is: \n")
    st.image(load_image(image_file),width=250)
    st.image(image_gr_28(load_image(image_file)),width=250)
    img= image_gr_28(load_image(image_file))
    # print(img.shape)
    img= np.invert(np.asarray(img))
    # print(img.shape)
    # prediction = model.predict(np.expand_dims(img[0],0))
    st.write(".......................  ____________________ .......................")
    path_in=image_file.name
    st.write(path_in)
    img2 = image_gr_28(load_image(image_file))

    img2= cv2.imread(img2)
    st.write(type(img2))
    img2 = np.invert(np.array([img2]))
    prediction2 = model.predict(img2)
    st.write(".......................  {}  .......................".format(np.argmax(prediction2)))
    # image_file =  image_file.read()
    # x_data = np.array( [np.array(cv2.imread(image_file[i])) for i in range(len(image_file))] )

    # pixels = x_data.flatten().reshape(1000, 12288)
    # print (pixels.shape)

image_number = st.sidebar.selectbox('Select the handwriten picture:',(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19))
try:
    # img2= cv2.imread(image_file)[:,:,0]
    # img2 = np.invert(np.array([img2]))
    # prediction2 = model.predict(img2)
    # st.write(".......................  {}  .......................".format(np.argmax(prediction2)))
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
# st.subheader('Prediction Probability')
# # st.write(prediction_proba)

# proba_df_clean = prediction.T
# proba_df= pd.DataFrame(proba_df_clean, columns=["Probabilities"])
# digits_n= ['0','1','2','3','4','5','6','7','8', '9']
# proba_df["Digits"]= digits_n
# # st.write(type(proba_df))
# column_names = ["Digits", "Probabilities"]
# proba_df = proba_df.reindex(columns=column_names)
# st.write(proba_df)
# fig = alt.Chart(proba_df).mark_bar().encode(x='Digits',y='Probabilities',color='Digits')
# st.altair_chart(fig,use_container_width=True)