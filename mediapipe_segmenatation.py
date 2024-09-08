import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

# Function to perform image segmentation and apply effects
def image_seg_enhance(input_img, bg_img=None, threshold=0.5, mode='gray'):
    img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    results = segment.process(img)
    binary_mask = results.segmentation_mask > threshold
    mask = np.dstack((binary_mask, binary_mask, binary_mask))
    ksize = 25
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if mode == 'gray':
        bg_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        bg_img_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2RGB)
        output_image = np.where(mask, img, bg_img_gray)
    elif mode == 'replace' and bg_img is not None:
        #print(input_img.size)
        bg_img = bg_img.resize(input_img.size)
        bg_img = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2BGR)
        output_image = np.where(mask, img, bg_img)
    elif mode == 'replace_gray' and bg_img is not None:
        #print("here", type(bg_img))
        #width, height = input_img.size
        #print(width, height)
        bg_img = bg_img.resize(input_img.size)
        bg_gray = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2GRAY)
        bg_img_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2RGB)
        output_image = np.where(mask, img, bg_img_gray)
    elif mode == 'blur':
        blurred_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        output_image = np.where(mask, img, blurred_img)
    elif mode == 'blur_gray':
        blurred_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        desat_blurred = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)
        desat_blurred = cv2.cvtColor(desat_blurred, cv2.COLOR_GRAY2RGB)
        output_image = np.where(mask, img, desat_blurred)
    else:
        print("here in last else")
        output_image = img
    return output_image


# Streamlit App
st.title("Segmentation with Effects")

result_img = None

with st.sidebar:
    # File uploader for the user to upload an image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    # Upload background image for replacement
    bg_file = st.file_uploader("Upload a background image (optional for replacement modes)", type=["jpg", "png", "jpeg"])

    # Threshold slider
    threshold = st.slider("Segmentation Threshold", min_value=0.0, max_value=1.0, value=0.5)

    # Select mode
    mode = st.selectbox(
        "Choose the effect:",
        ("gray", "replace", "replace_gray", "blur", "blur_gray")
    )

    if uploaded_file is not None:
        # Read the uploaded image
        input_img = Image.open(uploaded_file)
        
        # Display the original image
        st.image(input_img, caption="Original Image", use_column_width=True)

        # Perform segmentation and enhancement based on the selected mode
        # Replace or grayscale modes
        if bg_file:
            print("here")
            bg_img = Image.open(bg_file)
            print(bg_img.size)
        else:
            bg_img = None
        print(input_img.size)
        result_img = image_seg_enhance(input_img, bg_img=bg_img, threshold=threshold, mode=mode)

# Display the processed image
if isinstance(result_img, np.ndarray):
    st.image(result_img, caption="Processed Image", use_column_width=True)