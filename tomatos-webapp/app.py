import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load models
cnn_model = load_model("tomatos-webapp/CNN_TomatoModel.h5")
inception_model = load_model("tomatos-webapp/inceptionv3_Tomato.h5")

def getModel(model_option):
    # Select model
    if model_option == "CNN":
        st.sidebar.write("Selected model is CNN.")
        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return cnn_model
    elif model_option == "Inception":
        st.sidebar.write("Selected model is Inception.")
        inception_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return inception_model
    else:
        raise ValueError("Invalid model name")

# Define class labels
class_labels = ["Healthy", "Leaf Blight", "Leaf Curl", "Septoria Leaf Spot", "Verticillium Wilt"]
models = {
    "CNN": {"model": "CNN", "input_size": (150, 150)},
    "Inception": {"model": "Inception", "input_size": (224, 224)}
}
# App Title
st.title("Tomato Leaf Disease Classification")
st.subheader("Please upload the Tomato leaf image:")

# Sidebar: Model Selection
st.sidebar.title("Model Selection")

model_name = st.sidebar.selectbox("Choose a model:",list(models.keys()))
print("Model name is:",model_name)
selected_model = models.get(model_name)
model = None
model = getModel(model_name)
input_size = selected_model["input_size"]
print(input_size)

if model is None:
        st.error("Model not selected or defined. Please choose a valid model.")
        st.stop()  # Prevents further execution

# File Upload
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "png", "jpeg"])


 # Display the uploaded image
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize(input_size)  # Resize for compatibility with both models
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    print("uploaded images shape :",img_array.shape)
    print("Model's shape :",model.output_shape)
  
# Make predictions
    predictions = model.predict(img_array)
   
    print(predictions.shape)
    # Create DataFrame from predictions
    if predictions.shape[0] == 1:
       prediction_df = pd.DataFrame(predictions, columns=class_labels)
    else:  # Multiple predictions
      prediction_df = pd.DataFrame(predictions, columns=class_labels)
    
    predictions = predictions[0]
    print(predictions.shape)
    
    if predictions.ndim > 1:  # If still 2D
     aggregated_predictions = predictions.mean(axis=0)
    else:  # Already 1D
     aggregated_predictions = predictions
    
    print(f"Aggregated predictions: {aggregated_predictions}")

    predicted_class = np.argmax(aggregated_predictions)
   # Verify the type and value of predicted_class
    print(f"Predicted Class Type: {type(predicted_class)}")
    print(f"Predicted Class: {predicted_class}")


    
    # Display predictions
    st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
    st.write("**Prediction Probabilities:**")
    # prediction_df = pd.DataFrame(predictions, columns=class_labels)
    st.bar_chart(prediction_df.T)




test_path = "D:/Master/Reseach/Dataset-for-Crop-Pest-and-Disease-Detection/CCMT Dataset-Augmented/Tomato/test_set/"
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=input_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=True)



# Evaluate Metrics (Optional Dataset for Evaluation)
if st.sidebar.checkbox("Show Evaluation Metrics", value=False):
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_generator, batch_size=32)

    print(f"Testing Loss:{test_loss}")
    print(f"Testing Accuracy:{test_accuracy * 100:.4f}%")

    st.write(f"**Test Loss:** {test_loss:.4f}")
    st.write(f"**Test Accuracy:** {test_accuracy * 100:.2f}%")



 
