import streamlit as lit
import tensorflow as tf
from tensorflow import keras
import h5py
from PIL import Image, ImageOps
import numpy as np
import cv2


img = None
mainModel = None

def loadModel():
    mainModel = tf.keras.models.load_model('garbageModel.h5')
    mainModel.summary()

def predict(img, img2):
    mainModel = tf.keras.models.load_model('garbageModel.h5')

    img = img.resize((180,180))
    img = ImageOps.mirror(img)
    img = ImageOps.flip(img)



    img = np.array(img)

    #
    # img = cv2.imread(img2)
    # img = cv2.resize(img, (300,300))
    img = np.reshape(img, [1, 180, 180, 3])

    res = mainModel.predict(img)
    return res

def prepImage(img):
    img = img.resize((300,300))
    return img

#loadModel()

lit.title("Garbage Classification")
lit.text("Input an image below, and our program will identify which type of garbage it is: ")
lit.text("cardboard, glass, metal, paper, plastic, or trash")
img_file_buffer = lit.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

lit.title("Here is your selected image:")

if(img_file_buffer != None):
    img = Image.open(img_file_buffer)

if(img != None):
    lit.image(img)
    res = predict(img, img_file_buffer)
    print(res)
    for i,n in enumerate(res[0]):
        if(n == 1):
            if(i == 0):
                lit.text("Image is cardboard")
            if(i == 1):
                lit.text("Image is glass")
            if(i == 2):
                lit.text("Image is metal")
            if(i == 3):
                lit.text("Image is paper")
            if(i == 4):
                lit.text("Image is plastic")
            if(i == 5):
                lit.text("Image is trash")