import streamlit as st
import cv2
import os
from PIL import Image
import pandas as pd
import tensorflow as tf
from layers import L1Dist
import numpy as np
import datetime
import time
# Import uuid library to generate unique image names
import uuid
from csv import DictWriter
import csv


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():

    sidebar = st.sidebar.selectbox(
        'Choose one of the following', ('Welcome', 'Step 1: Register as new user', 'Step 2: Capture Input Images', 'Step 3: Face Recognition'))

    # load the model
    model = tf.keras.models.load_model(
        'model.h5', custom_objects={'L1Dist': L1Dist})

    if sidebar == 'Welcome':
        st.title('COS30082 Applied Machine Learning')
        st.header("Group 3: Face recognition attendance system")
        st.image('app_images/welcome.jpg', use_column_width=True)
        st.subheader('Team members:')
        st.text('1. Clement Goh Yung Jing (101218668) \n2. Lee Zhe Sheng (10215371)\n3. Cheryl Tan Shen Wey (101222753) \n4. Vibatha Naveen Jayakody Jayakody Arachcilage (101232163)')

    if sidebar == 'Step 1: Register as new user':

        st.title('Register as new user')

        st.text(
            'Its good to have at least 5 images for initialising your identity.')
        register_btn = st.button('Register as new user')

        st.caption('Please check the checkbox below for web cam previewing')

        run = st.checkbox('Webcam Preview')

        webcam_preview(run)

        if register_btn:
            with st.spinner('Registering...'):
                REGISTER_PATH = os.path.join(
                    'application_data', 'verification_images')
                camera = cv2.VideoCapture(0)
                REGISTER_PATH2 = os.path.join(
                    REGISTER_PATH, '{}.jpg'.format(uuid.uuid1()))
                ret, frame = camera.read()
                cv2.imwrite(REGISTER_PATH2, frame)

                st.image(REGISTER_PATH2)

            st.success('Register Successfully!')

    if sidebar == 'Step 2: Capture Input Images':
        st.title('Capture Input Image')

        st.caption('Press the button again to capture again')
        capture_btn = st.button('Capture Input Image')

        if capture_btn:
            capture_input()

    if sidebar == 'Step 3: Face Recognition':

        st.title("Face Recognition (Web Cam)")

        # Slider for threshold
        # Detection Threshold
        st.subheader('Detection Threshold')
        detection_threshold = st.slider(
            'Try out threshold between 0-1', value=0.53)
        st.write("Detection Threshold:", detection_threshold)

        # Verification Threshold
        st.subheader('Verification Threshold')
        verification_threshold = st.slider(
            'Try out threshold between 0-1', value=0.59)
        st.write("Verification Threshold:", verification_threshold)

        st.subheader('Step 3')
        step_3(model, detection_threshold, verification_threshold)


def capture_input():
    with st.spinner('Capturing...'):
        camera = cv2.VideoCapture(0)
        SAVE_PATH = os.path.join(
            'application_data', 'input_image', 'input_image.jpg')
        ret, frame = camera.read()
        cv2.imwrite(SAVE_PATH, frame)
        st.image('application_data/input_image/input_image.jpg',
                 use_column_width=True)
    st.success('Done!')


def verification(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join(
            'application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join(
            'application_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(
            list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / \
        len(os.listdir(os.path.join(
            'application_data', 'verification_images')))
    verified = verification > verification_threshold

    st.text(
        f"Result of comparison with \nfaces registered in database: {results}")
    st.text(f"Number of predictions: {detection}")
    st.text(f"Verification score: {verification}")
    st.text(f"Verification result: {verified}")
    st.text('Verified' if verified == True else 'Unverified')

    return verified


def step_3(model, detection_threshold, verification_threshold):

    username = st.text_input("Enter you name")
    col1, col2 = st.columns(2)

    with col1:
        clockIn_btn = st.button("Clock In")
        if clockIn_btn:
            with st.spinner('Verifying...'):
                verified = verification(
                    model, detection_threshold, verification_threshold)
                if verified == True:
                    clocking(username, "Clocked In")
                    st.success(f"{username} Clocked In")
                else:
                    st.warning('User is not verified')

            st.success(f"Done!")

    with col2:
        clockOut_btn = st.button("Clock Out")
        if clockOut_btn:
            with st.spinner('Verifying...'):
                verified = verification(
                    model, detection_threshold, verification_threshold)
                if verified == True:
                    clocking(username, "Clocked Out")
                    st.success(f"{username} Clocked Out")
                else:
                    st.warning('User is not verified')
            st.success(f"Done!")


def clocking(username, mode):

    currentDate = datetime.date.today()
    currentTime_in = datetime.datetime.now().time()
    fieldNames = ["Name", "Date", "Time", "Mode"]

    with open('data.csv', 'a', newline='') as Clock:
        Clockin_writer = csv.writer(Clock)
        #Clockin_writer.writerow({"Name":username, "Date":currentDate, "Time":currentTime_in, "Mode":mode})
        Clockin_writer.writerow([username, currentDate, currentTime_in, mode])

# Load image from file and convert to 105x105 pixel


def preprocess(file_path):

    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (105, 105))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img


def webcam_preview(run):

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)


if __name__ == "__main__":
    main()
