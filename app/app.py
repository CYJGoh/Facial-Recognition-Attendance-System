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
import csv
from csv import DictWriter


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


path = 'application_data/verification_images'


def main():

    sidebar = st.sidebar.selectbox(
        'Choose one of the following', ('Welcome', 'Step 1: Register as new user', 'Step 2: Capture Input Images', 'Step 3: Face Recognition'))

    # load the model
    model = tf.keras.models.load_model(
        'model.h5', custom_objects={'L1Dist': L1Dist}, compile=False)

    if sidebar == 'Welcome':
        st.title('COS30082 Applied Machine Learning')
        st.header("Group 3: Face recognition attendance system")
        st.image('app_images/welcome.jpg', use_column_width=True)
        st.subheader('Team members:')
        st.text('1. Clement Goh Yung Jing (101218668) \n2. Lee Zhe Sheng (10215371)\n3. Cheryl Tan Shen Wey (101222753) \n4. Vibatha Naveen Jayakody Jayakody Arachcilage (101232163)')

    if sidebar == 'Step 1: Register as new user':

        st.title('Step 1: Register as new user')

        st.caption('Please check the checkbox below for web cam previewing')

        # run = st.checkbox('Webcam Preview')

        # webcam_preview(run)

        director_name = st.text_input("Please Enter you name")

        register_btn = st.button('Register as new user')

        if director_name == '':
            st.warning('Do not empty the input field...')

        if register_btn:
            with st.spinner('Registering...'):

                camera = cv2.VideoCapture(0)

                REGISTER_PATH = os.path.join(
                    'application_data', 'verification_images', director_name)

                try:
                    os.makedirs(REGISTER_PATH, exist_ok=True)

                    REGISTER_PATH2 = os.path.join(
                        REGISTER_PATH, '{}.jpg'.format(uuid.uuid1()))
                    ret, frame = camera.read()
                    cv2.imwrite(REGISTER_PATH2, frame)

                    st.image(REGISTER_PATH2)

                    st.success("User Image Registered Successfully !")

                except OSError as error:
                    st.error("User Register Unsuccessfully! Please try it again")

    if sidebar == 'Step 2: Capture Input Images':
        st.title('Step 2: Capture Input Image')

        st.caption('Press the button again to capture again')
        capture_btn = st.button('Capture Input Image')

        if capture_btn:
            capture_input()

    if sidebar == 'Step 3: Face Recognition':

        st.title("Step 3: Face Recognition (Web Cam)")

        detection_threshold = 0.53

        verification_threshold = 0.59

        st.caption(
            'Please choose your action by selecting the action button below: ')
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


def verification_clocked_in(model, detection_threshold, verification_threshold):

    for root, directories, files in os.walk(path, topdown=False, followlinks=True):

        for directory in directories:

            results = []

            for file in os.listdir(os.path.join(root, directory)):
                if os.path.isfile(os.path.join(root, directory, file)):
                    input_img = preprocess(os.path.join(
                        'application_data', 'input_image', 'input_image.jpg'))
                    validation_img = preprocess(
                        os.path.join(root, directory, file))

                # # Time tracking for image prediction
                # prediction_time = time.process_time()

                # Make Predictions
                result = model.predict(
                    list(np.expand_dims([input_img, validation_img], axis=1)))
                results.append(result)

                # st.text(
                #     f"Prediction Time: {time.process_time() - prediction_time}")

                # Detection Threshold: Metric above which a prediciton is considered positive
                detection = np.sum(np.array(results) > detection_threshold)

                # Verification Threshold: Proportion of positive predictions / total positive samples
                verification = detection / \
                    len(os.listdir(os.path.join(
                        path, directory)))
                verified = verification > verification_threshold

            if verified == True:
                clocking(directory, "Clocked In")
                st.success(
                    f"Welcome Back {directory}, You are Clocked In Successfully!")

                break

            else:
                continue


def verification_clocked_out(model, detection_threshold, verification_threshold):

    for root, directories, files in os.walk(path, topdown=False, followlinks=True):

        for directory in directories:

            results = []

            for file in os.listdir(os.path.join(root, directory)):
                if os.path.isfile(os.path.join(root, directory, file)):
                    input_img = preprocess(os.path.join(
                        'application_data', 'input_image', 'input_image.jpg'))
                    validation_img = preprocess(
                        os.path.join(root, directory, file))

                # # Time tracking for image prediction
                # prediction_time = time.process_time()

                # Make Predictions
                result = model.predict(
                    list(np.expand_dims([input_img, validation_img], axis=1)))
                results.append(result)

                # st.text(
                #     f"Prediction Time for {directory}: {time.process_time() - prediction_time}")

                # Detection Threshold: Metric above which a prediciton is considered positive
                detection = np.sum(np.array(results) > detection_threshold)

                # Verification Threshold: Proportion of positive predictions / total positive samples
                verification = detection / \
                    len(os.listdir(os.path.join(
                        path, directory)))
                verified = verification > verification_threshold

            if verified == True:
                clocking(directory, "Clocked Out")
                st.success(
                    f"Have a Nice Day {directory}, You are Clocked Out Successfully!")

                break

            else:
                continue


def step_3(model, detection_threshold, verification_threshold):

    col1, col2 = st.columns(2)

    with col1:
        clockIn_btn = st.button("Clock In")
        if clockIn_btn:
            with st.spinner('Verifying...'):
                verification_clocked_in(
                    model, detection_threshold, verification_threshold)

    with col2:
        clockOut_btn = st.button("Clock Out")
        if clockOut_btn:
            with st.spinner('Verifying...'):
                verification_clocked_out(
                    model, detection_threshold, verification_threshold)


def clocking(username, mode):

    currentDate = datetime.date.today()
    currentTime_in = datetime.datetime.now().time()
    fieldNames = ["Name", "Date", "Time", "Mode"]

    with open('data.csv', 'a', newline='') as Clock:
        Clockin_writer = csv.writer(Clock)
        # Clockin_writer.writerow({"Name":username, "Date":currentDate, "Time":currentTime_in, "Mode":mode})
        Clockin_writer.writerow([username, currentDate, currentTime_in, mode])

# Load image from file and convert to 105x105 pixel


def preprocess(file_path):

    # Read in image from file path

    byte_img = tf.io.read_file(file_path)

    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 105x105x3
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
