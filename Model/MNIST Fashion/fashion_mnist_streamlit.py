"""
Fashion-MNIST Recognition with Streamlit

Interactive web interface for image classification using trained model.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st
import sys

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Model.model import Model


MODEL_PATH = Path(__file__).parent / "fashion_mnist.params"
TEST_IMAGES_DIR = Path(__file__).parent / "Fashion_Mnist_Images" / "test"

FASHION_MNIST_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


@st.cache_resource
def load_trained_model(model_path: str) -> Model:
    """Load trained model with caching."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        st.info("Please train model first using fashion_mnist_train.py")
        st.stop()
    
    try:
        model = Model.load(str(model_path))
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()


def load_image(image_path: str) -> np.ndarray:
    """Load single image and convert to grayscale."""
    try:
        img = Image.open(image_path)
        
        if img.mode != 'L':
            img = img.convert('L')
        
        if img.size != (28, 28):
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        return np.array(img, dtype=np.uint8)
    
    except Exception as e:
        st.error(f"Failed to load image {image_path}: {str(e)}")
        return None


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference."""
    image = image.astype(np.float32) / 255.0
    image = image.reshape(-1)
    return image


def predict_single_image(model: Model, image: np.ndarray) -> tuple:
    """Predict class for single image."""
    X = image.reshape(1, -1)
    logits = model.forward(X, training=False)
    probs = logits[0]
    
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    
    return predicted_class, confidence, probs


def find_test_images(directory: str) -> list:
    """Find all test images in directory."""
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(set(image_files))


def main():
    st.set_page_config(page_title="Fashion-MNIST Classifier", layout="wide")
    
    st.title("Fashion-MNIST Image Classifier")
    
    st.write("Upload images or select from test images to classify clothing and accessories.")
    
    model = load_trained_model(MODEL_PATH)
    
    tab1, tab2 = st.tabs(["Upload Image", "Test Images"])
    
    with tab1:
        st.header("Upload Single Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"]
        )
        
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file)
            
            if raw_image.mode != 'L':
                raw_image = raw_image.convert('L')
            
            if raw_image.size != (28, 28):
                raw_image = raw_image.resize((28, 28), Image.Resampling.LANCZOS)
            
            raw_array = np.array(raw_image, dtype=np.uint8)
            preprocessed = preprocess_image(raw_array)
            
            predicted_class, confidence, probs = predict_single_image(model, preprocessed)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(raw_array, caption="Input Image", use_column_width=True, clamp=True)
            
            with col2:
                st.metric(
                    "Predicted Class",
                    FASHION_MNIST_LABELS[predicted_class]
                )
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )
            
            st.write("Class Probabilities:")
            
            probs_dict = {
                f"[{i}] {FASHION_MNIST_LABELS[i]}": float(probs[i])
                for i in range(10)
            }
            
            st.bar_chart(probs_dict)
    
    with tab2:
        st.header("Test Images")
        
        image_files = find_test_images(str(TEST_IMAGES_DIR))
        
        if not image_files:
            st.warning(f"No test images found in {TEST_IMAGES_DIR}")
            st.info("Create subdirectories 0-9 in Fashion_Mnist_Images/test/ and add images")
        else:
            st.write(f"Found {len(image_files)} test images")
            
            selected_image = st.selectbox(
                "Select an image",
                image_files,
                format_func=lambda x: x.name
            )
            
            if selected_image:
                raw_image = load_image(str(selected_image))
                
                if raw_image is not None:
                    preprocessed = preprocess_image(raw_image)
                    predicted_class, confidence, probs = predict_single_image(model, preprocessed)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(raw_image, caption=selected_image.name, use_column_width=True)
                    
                    with col2:
                        st.metric(
                            "Predicted Class",
                            FASHION_MNIST_LABELS[predicted_class]
                        )
                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%"
                        )
                    
                    st.write("Class Probabilities:")
                    
                    probs_dict = {
                        f"[{i}] {FASHION_MNIST_LABELS[i]}": float(probs[i])
                        for i in range(10)
                    }
                    
                    st.bar_chart(probs_dict)
    
    st.sidebar.write("Fashion-MNIST Classes:")
    for class_idx, class_name in FASHION_MNIST_LABELS.items():
        st.sidebar.write(f"{class_idx}: {class_name}")


if __name__ == "__main__":
    main()
