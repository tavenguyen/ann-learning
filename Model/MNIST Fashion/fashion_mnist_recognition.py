"""
Fashion-MNIST Recognition & Visualization
===========================================

Load trained model and perform predictions on Fashion-MNIST test images.
Includes visualization of predictions with confidence scores.

This script demonstrates model inference and image preprocessing pipelines.
"""

import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path to import framework
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Model.model import Model


# ==================== Constants ====================

MODEL_PATH = Path(__file__).parent / "fashion_mnist.params"
TEST_IMAGES_DIR = Path(__file__).parent / "Fashion_Mnist_Images" / "test"

# Fashion-MNIST class labels
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

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


# ==================== Model Loading ====================

def load_trained_model(model_path: str) -> Model:
    """
    Load trained Fashion-MNIST classifier model.
    
    Args:
        model_path: Path to saved model parameters
    
    Returns:
        Loaded Model instance ready for prediction
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Please train model first using fashion_mnist_train.py"
        )
    
    try:
        model = Model.load(str(model_path))
        print(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


# ==================== Image Preprocessing ====================

def load_image(image_path: str) -> np.ndarray:
    """
    Load single image and convert to grayscale if needed.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Grayscale image array (28, 28)
    
    Raises:
        Exception: If image cannot be loaded or resized
    """
    try:
        img = Image.open(image_path)
        
        # Convert to grayscale if necessary
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28
        if img.size != (28, 28):
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        return np.array(img, dtype=np.uint8)
    
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    MUST be identical to training preprocessing:
    - Normalize to [0, 1]
    - Convert to float32
    - Flatten to (784,)
    
    Args:
        image: Grayscale image (28, 28) as uint8
    
    Returns:
        Preprocessed feature vector (784,)
    """
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Flatten
    image = image.reshape(-1)
    
    return image


def load_and_preprocess_image(image_path: str) -> tuple:
    """
    Load and preprocess single image for prediction.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (preprocessed_feature_vector, original_image_array)
    """
    raw_image = load_image(image_path)
    preprocessed = preprocess_image(raw_image)
    return preprocessed, raw_image


# ==================== Prediction ====================

def predict_single_image(model: Model, image: np.ndarray) -> tuple:
    """
    Predict class for single image.
    
    Args:
        model: Trained model
        image: Preprocessed feature vector (784,)
    
    Returns:
        Tuple of (predicted_class_index, confidence)
    """
    # Add batch dimension
    X = image.reshape(1, -1)
    
    # Forward pass (no training mode, just inference)
    logits = model.forward(X, training=False)
    
    # Get probabilities from softmax output
    probs = logits[0]
    
    # Get class and confidence
    predicted_class = np.argmax(probs)
    confidence = float(np.max(probs))
    
    return int(predicted_class), confidence


def predict_batch(model: Model, images: np.ndarray) -> tuple:
    """
    Predict classes for batch of images.
    
    Args:
        model: Trained model
        images: Batch of preprocessed images (N, 784)
    
    Returns:
        Tuple of (predicted_classes, confidences)
            - predicted_classes: array of shape (N,)
            - confidences: array of shape (N,)
    """
    logits = model.forward(images, training=False)
    predicted_classes = np.argmax(logits, axis=1)
    confidences = np.max(logits, axis=1)
    
    return predicted_classes, confidences


# ==================== Visualization ====================

def display_prediction(image: np.ndarray, predicted_class: int,
                       confidence: float, filename: str = ""):
    """
    Display image with prediction using matplotlib.
    
    Args:
        image: Original grayscale image (28, 28)
        predicted_class: Predicted class index
        confidence: Prediction confidence (0-1)
        filename: Optional filename to display in window
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Display image
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    
    # Create title with prediction
    label_name = FASHION_MNIST_LABELS[predicted_class]
    confidence_pct = confidence * 100
    
    title = f"{label_name} ({confidence_pct:.1f}%)"
    if filename:
        title = f"{filename}\n{title}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def print_prediction_result(filename: str, predicted_class: int, confidence: float):
    """
    Print prediction result in formatted way.
    
    Args:
        filename: Image filename
        predicted_class: Predicted class index
        confidence: Prediction confidence (0-1)
    """
    label_name = FASHION_MNIST_LABELS[predicted_class]
    confidence_pct = confidence * 100
    
    print(f"  {filename:30s} -> [{predicted_class}] {label_name:15s} ({confidence_pct:5.1f}%)")


# ==================== File Discovery ====================

def find_test_images(directory: str) -> list:
    """
    Find all test images in directory.
    
    Args:
        directory: Path to directory containing test images
    
    Returns:
        Sorted list of image file paths
    
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(
            f"Test images directory not found: {directory}\n"
            f"Expected structure:\n"
            f"  Model/MNIST Fashion/Fashion_Mnist_Images/test/*.jpg"
        )
    
    # Find all supported image files
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))  # Remove duplicates and sort
    
    return image_files


# ==================== Batch Inference ====================

def process_test_images(model: Model, test_dir: str, visualize: bool = True,
                        max_images: int = None) -> dict:
    """
    Process all test images and generate predictions.
    
    Args:
        model: Trained model
        test_dir: Directory containing test images
        visualize: Whether to show matplotlib visualization
        max_images: Maximum number of images to process (None = all)
    
    Returns:
        Dictionary with prediction results and statistics
    """
    # Find test images
    try:
        image_files = find_test_images(test_dir)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return None
    
    if not image_files:
        print(f"Warning: No test images found in {test_dir}")
        return None
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\nFound {len(image_files)} test image(s)")
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)
    
    results = {
        'filenames': [],
        'predictions': [],
        'confidences': [],
        'labels': []
    }
    
    for image_path in image_files:
        try:
            # Load and preprocess
            preprocessed, raw_image = load_and_preprocess_image(str(image_path))
            
            # Predict
            predicted_class, confidence = predict_single_image(model, preprocessed)
            
            # Store results
            results['filenames'].append(image_path.name)
            results['predictions'].append(predicted_class)
            results['confidences'].append(confidence)
            results['labels'].append(FASHION_MNIST_LABELS[predicted_class])
            
            # Print result
            print_prediction_result(image_path.name, predicted_class, confidence)
            
            # Visualize if requested
            if visualize:
                display_prediction(raw_image, predicted_class, confidence, image_path.name)
        
        except Exception as e:
            print(f"  Error processing {image_path.name}: {str(e)}")
            continue
    
    print("="*70)
    
    return results


def print_statistics(results: dict):
    """
    Print statistics about predictions.
    
    Args:
        results: Prediction results dictionary
    """
    if not results or not results['predictions']:
        return
    
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total images processed: {len(results['predictions'])}")
    print(f"\nAverage confidence: {np.mean(results['confidences']):.4f}")
    print(f"Min confidence: {np.min(results['confidences']):.4f}")
    print(f"Max confidence: {np.max(results['confidences']):.4f}")
    
    # Class distribution
    from collections import Counter
    class_counts = Counter(results['predictions'])
    
    print(f"\nPrediction distribution:")
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        pct = (count / len(results['predictions'])) * 100
        label = FASHION_MNIST_LABELS[class_idx]
        print(f"  [{class_idx}] {label:15s}: {count:3d} ({pct:5.1f}%)")
    
    print("="*70)


# ==================== Main ====================

def main(visualize: bool = True, max_images: int = None):
    """
    Main recognition pipeline.
    
    Args:
        visualize: Whether to display predictions visually
        max_images: Maximum number of images to process (useful for testing)
    """
    try:
        # Load model
        print("\nLoading trained model...")
        model = load_trained_model(MODEL_PATH)
        
        # Process test images
        print(f"\nProcessing test images from {TEST_IMAGES_DIR}")
        results = process_test_images(model, str(TEST_IMAGES_DIR),
                                      visualize=visualize, max_images=max_images)
        
        if results:
            # Print statistics
            print_statistics(results)
        
        print("\nRecognition pipeline completed successfully!")
        
        return results
    
    except Exception as e:
        print(f"Error in recognition pipeline: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run recognition pipeline
    # Set visualize=False if running in headless environment
    results = main(visualize=True, max_images=None)
