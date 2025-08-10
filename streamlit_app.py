import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image    # For image loading and preprocessing
import numpy as np
from PIL import Image                               # Pillow library for image manipulation
import os                                           # For path management
import base64                                       # Base64 encoding
import io                                           # handling image bytes

def get_base64_image(image_path):
    
    #Encodes a local image file to a Base64 string.
    
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    
    except FileNotFoundError:
        st.error(f"Error: Background image file not found at {image_path}. Please check the path.")
        return None
    
    except Exception as e:
        st.error(f"Error encoding background image: {e}")
        return None
    
def set_custom_background():

    # Option 1: Image Background (use with caution for medical apps - keep it subtle!)
    LOCAL_IMAGE_PATH = "background.jpg" 
    base64_image = get_base64_image(LOCAL_IMAGE_PATH)

    if base64_image:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{base64_image}"); /* Assumes PNG, change if different */
                background-size: cover; /* or 'contain' or specify dimensions */
                background-repeat: no-repeat; /* or 'repeat' */
                background-attachment: fixed; /* Keeps background fixed when scrolling */
                opacity: 0.8; /* Make it slightly transparent so text is readable */
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    # Option 2: A Professional, Subtle Light Blue Gradient (Recommended for this project)
    #st.markdown(
     #   """
      #  <style>
       # .stApp {
        #    background: linear-gradient(to bottom, #f0f8ff, #e6f7ff); /* Very light blue to slightly darker light blue */
        #}
        #</style>
        #""",
        #unsafe_allow_html=True
    #)


set_custom_background()

MODEL_PATH = 'best_custom_cnn_model.keras' # Assuming it's in the same directory as this script

IMG_HEIGHT = 224 # Must match the input size your model was trained with
IMG_WIDTH = 224  # Must match the input size your model was trained with
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary'] # Must match your training classes

# --- Load the trained model ---
# Use st.cache_resource to load the model only once, improving app performance.
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model is in the correct directory.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Prediction Function ---
def predict_tumor_type(img_path, model):
    
    if model is None:
        return "Model not loaded.", 0.0, []

    try:
        # Load and resize the image
        img = Image.open(img_path).resize((IMG_HEIGHT, IMG_WIDTH))
        
        # Convert to numpy array
        img_array = image.img_to_array(img)
        
        # Ensure 3 channels (even for grayscale, as model expects 3)
        if img_array.shape[-1] == 1: # If grayscale (H, W, 1)
            img_array = np.stack([img_array[:,:,0]]*3, axis=-1) # Duplicate channel to 3
        
        # Add a batch dimension (model expects input in batches: (1, H, W, C))
        img_array = np.expand_dims(img_array, axis=0)
        
        # The model should handle its own preprocessing (Rescaling, preprocess_input)
        # as defined in your Keras model architecture.
        # If your model's input layers do NOT include Rescaling(1./255) or preprocess_input,
        # you would add them here:
        # img_array = img_array / 255.0 # Example if normalization is not in model

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get the predicted class index and confidence
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100 # Convert to percentage
        
        predicted_class_name = CLASSES[predicted_class_index]
        
        # Get all class probabilities for display
        all_class_confidences = {CLASSES[i]: predictions[0][i] * 100 for i in range(len(CLASSES))}

        return predicted_class_name, confidence, all_class_confidences

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction failed.", 0.0, {}

# --- Streamlit UI ---
st.set_page_config(
    page_title="NeuroScan AI: Precision Brain Tumor Insight",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🧠 NeuroScan AI: Precision Brain Tumor Insight")
st.markdown("Upload a brain MRI image to get a prediction on the tumor type.")

st.markdown("""
    This application uses a deep learning model (Custom CNN) to classify brain MRI images into one of four categories:
    - **Glioma Tumor**
    - **Meningioma Tumor**
    - **No Tumor**
    - **Pituitary Tumor**
    
    The model was trained on a comprehensive dataset of brain MRI scans.
""")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a Brain MRI image...",
    type=["jpg", "jpeg", "png", "bmp", "tiff"]
)

if uploaded_file is not None:
    
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    predicted_type, confidence, all_confidences = predict_tumor_type(uploaded_file, model)

    if predicted_type != "Prediction failed.":
        st.subheader(f"Prediction: {predicted_type}")
        st.write(f"Confidence: {confidence:.2f}%")

        st.markdown("---")
        st.subheader("All Class Probabilities:")
        
        # Sort probabilities for better display
        sorted_confidences = sorted(all_confidences.items(), key=lambda item: item[1], reverse=True)
        
        for class_name, prob in sorted_confidences:
            st.write(f"**{class_name}:** {prob:.2f}%")
            st.progress(float(prob) / 100) # Display a progress bar for confidence

        if confidence < 70: # Example threshold for low confidence
            st.warning("Low confidence prediction. Consider consulting a medical professional for diagnosis.")
        elif confidence > 95 and predicted_type != "No Tumor":
            st.info("High confidence prediction. Please note this is an AI-based prediction and not a substitute for professional medical advice.")
        elif confidence > 95 and predicted_type == "No Tumor":
            st.success("High confidence prediction. The image likely shows no tumor.")

    else:
        st.error("Could not make a prediction. Please try another image or check the model path.")

else:
    st.info("Please upload an MRI image to get a tumor classification.")
    st.markdown("---")
    st.markdown("### How to Use:")
    st.markdown("1. Click 'Browse files' above.")
    st.markdown("2. Select a brain MRI image file (JPG, PNG, etc.).")
    st.markdown("3. The app will automatically classify the image and display the results.")

st.markdown("---")
st.caption("Disclaimer: This application is for educational and demonstrative purposes only and should not be used for medical diagnosis. Always consult with a qualified healthcare professional.")

