"""
Fashion-MNIST Training Pipeline
================================

A production-quality ANN classifier for Fashion-MNIST using custom neural network framework.
Demonstrates best practices in deep learning engineering.

Architecture: 784 -> 512 -> 256 -> 128 -> 64 -> 10
- Batch normalization through careful weight initialization
- L2 regularization to prevent overfitting
- Adam optimizer with learning rate decay
- Dropout for regularization
- Mini-batch gradient descent with early stopping criterion

Fashion-MNIST Classes:
    0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
    5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
"""

import numpy as np
import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import framework
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Model.model import (
    Model,
    DenseLayer,
    Activation_LeakyReLU,
    Layer_Dropout,
    Activation_Softmax_Loss_CategoricalCrossEntropy,
    Optimizer_Adam,
    Accuracy_CategoricalClassification,
    Regularization_L2,
    DataLoader,
)


# ==================== Constants ====================

RANDOM_SEED = 42
N_CLASSES = 10
IMAGE_SIZE = 28
INPUT_FEATURES = IMAGE_SIZE * IMAGE_SIZE  # 784
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

MODEL_SAVE_PATH = Path(__file__).parent / "fashion_mnist.params"


# ==================== Data Loading & Preprocessing ====================

def load_fashion_mnist_from_local():
    """
    Load Fashion-MNIST dataset from local directories.
    Expected structure: Fashion_Mnist_Images/train/ and Fashion_Mnist_Images/test/
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    from PIL import Image
    
    train_dir = Path(__file__).parent / "Fashion_Mnist_Images" / "train"
    test_dir = Path(__file__).parent / "Fashion_Mnist_Images" / "test"
    
    if not train_dir.exists():
        raise RuntimeError(f"Train directory not found: {train_dir}")
    if not test_dir.exists():
        raise RuntimeError(f"Test directory not found: {test_dir}")
    
    X_train, y_train = load_images_from_directory(train_dir)
    X_test, y_test = load_images_from_directory(test_dir)
    
    return X_train, y_train, X_test, y_test


def load_images_from_directory(directory):
    """
    Load all images from directory. Expects subdirectories for each class (0-9).
    
    Returns:
        Tuple of (images_array, labels_array)
    """
    from PIL import Image
    
    images = []
    labels = []
    total_loaded = 0
    
    print(f"\nLoading images from {directory}...")
    
    for class_idx in range(10):
        class_dir = Path(directory) / str(class_idx)
        if not class_dir.exists():
            continue
            
        image_files = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
        print(f"  Class {class_idx}: Found {len(image_files)} images", end="")
        
        class_loaded = 0
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                if img.mode != 'L':
                    img = img.convert('L')
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                images.append(np.array(img, dtype=np.uint8))
                labels.append(class_idx)
                class_loaded += 1
                total_loaded += 1
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {str(e)}")
        
        print(f" - Loaded {class_loaded}")
    
    if not images:
        raise RuntimeError(f"No images found in {directory}")
    
    print(f"Total images loaded: {total_loaded}\n")
    
    return np.array(images), np.array(labels, dtype=np.uint8)


def load_dataset():
    """
    Load Fashion-MNIST dataset from local directories.
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    
    Raises:
        RuntimeError: If dataset cannot be loaded
    """
    try:
        X_train, y_train, X_test, y_test = load_fashion_mnist_from_local()
        print(f"Loaded dataset: train={X_train.shape}, test={X_test.shape}")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")


def preprocess_images(X: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Normalize images to [0, 1] range and convert dtype.
    
    Args:
        X: Image array (N, 28, 28) with uint8 values
        dtype: Target data type
    
    Returns:
        Normalized array (N, 784) as float32
    """
    X = X.astype(dtype) / 255.0  # Normalize to [0, 1]
    return X.reshape(X.shape[0], -1)  # Flatten to (N, 784)


def split_train_validation(X: np.ndarray, y: np.ndarray, split: float = 0.2,
                           random_state: int = None) -> tuple:
    """
    Split training data into train and validation sets.
    
    Args:
        X: Features (N, D)
        y: Labels (N,)
        split: Validation fraction (0.0 to 1.0)
        random_state: Seed for reproducibility
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_val = int(n_samples * split)
    
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def smooth_labels(y, alpha=0.1):
    """
    Apply label smoothing to convert hard labels to soft labels.
    
    Args:
        y: Labels as a 1D array of class indices.
        alpha: Smoothing factor (0.0 to 1.0).
        
    Returns:
        Soft labels as a 2D array (N, N_CLASSES).
    """
    n_samples = len(y)
    n_classes = N_CLASSES
    
    # Create one-hot encoding
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y] = 1.0
    
    # Apply smoothing: (1 - alpha) * one_hot + alpha / n_classes
    return (1.0 - alpha) * one_hot + (alpha / n_classes)


def prepare_dataset():
    """
    Load and preprocess Fashion-MNIST dataset.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Load raw data
    X_train, y_train, X_test, y_test = load_dataset()
    
    # Preprocess
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)
    
    # Split validation
    X_train, y_train, X_val, y_val = split_train_validation(
        X_train, y_train,
        split=VALIDATION_SPLIT,
        random_state=RANDOM_SEED
    )
    
    # Apply Label Smoothing to training and validation sets
    y_train_smooth = smooth_labels(y_train, alpha=0.1)
    y_val_smooth = smooth_labels(y_val, alpha=0.1)
    
    print("Dataset prepared:")
    print(f"  Train: {X_train.shape}, {y_train_smooth.shape}")
    print(f"  Validation: {X_val.shape}, {y_val_smooth.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train_smooth, X_val, y_val_smooth, X_test, y_test


# ==================== Model Architecture ====================

def build_model() -> Model:
    """
    Build Fashion-MNIST classifier architecture.
    
    Design rationale:
    - Input layer: 784 features (28x28 images flattened)
    - Hidden layers: 512 -> 256 -> 128 -> 64 neurons (progressively smaller)
    - Activation: LeakyReLU for non-linearity
    - Dropout: 30% after each hidden layer to prevent overfitting
    - Regularization: L2 (0.0001) on dense layer weights
    - Output: 10 classes with Softmax + Cross-Entropy loss
    
    Rationale for this architecture:
    - Fashion-MNIST is a 10-class image classification task
    - Progressive width reduction reduces parameters and improves learning
    - L2 regularization prevents weight explosion
    - Dropout improves generalization
    - Adam optimizer handles learning rate adaptation
    
    Returns:
        Compiled Model ready for training
    """
    model = Model()
    
    # Input: 784 features
    # Hidden layer 1: 256 neurons
    reg = Regularization_L2(strength=1e-3)
    model.add(DenseLayer(256, INPUT_FEATURES, weight_regularizer=reg, init_type="he"))
    model.add(Activation_LeakyReLU())
    model.add(Layer_Dropout(dropout_rate=0.3))
    
    # Hidden layer 2: 256 -> 128 neurons
    model.add(DenseLayer(128, 256, weight_regularizer=reg, init_type="he"))
    model.add(Activation_LeakyReLU())
    model.add(Layer_Dropout(dropout_rate=0.3))

    # Output layer: 10 classes
    model.add(DenseLayer(N_CLASSES, 128))
    
    # Optimizer: Adam with learning rate decay
    optimizer = Optimizer_Adam(
        learning_rate=LEARNING_RATE,
        decay_rate=1e-5  # Adjusted decay rate from 5e-4 to 1e-5
    )
    
    # Loss + Activation: Softmax + Categorical Cross-Entropy
    loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
    
    # Accuracy metric
    accuracy = Accuracy_CategoricalClassification()
    
    # Compile model
    model.set(loss=loss, optimizer=optimizer, accuracy=accuracy)
    model.finalize()
    
    return model


# ==================== Training ====================

def train_model(model: Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Train model with validation monitoring and best weight saving.
    
    Training strategy:
    - Mini-batch gradient descent (batch size 128)
    - Validation every epoch
    - Save model only when validation accuracy improves
    - Early stopping criteria: if validation accuracy doesn't improve for 15 epochs
    
    Args:
        model: Compiled neural network model
        X_train: Training features (N, 784)
        y_train: Training labels (N,)
        X_val: Validation features (M, 784)
        y_val: Validation labels (M,)
    
    Returns:
        Dictionary with training history and best metrics
    """
    print("\n" + "="*70)
    print("TRAINING FASHION-MNIST CLASSIFIER")
    print("="*70)
    print(f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE} | L2 regularization: 1e-4")
    print(f"Dropout: 20-30% | Architecture: 784->512->256->128->64->10")
    print("="*70)
    
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    early_stop_patience = 15
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch': []
    }
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Reset accuracy for new epoch
        if model.accuracy is not None:
            model.accuracy.new_pass()
        
        epoch_data_loss = 0
        epoch_reg_loss = 0
        
        # Mini-batch training loop
        train_loader = DataLoader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model.forward(X_batch, training=True)
            
            # Calculate loss
            data_loss = model.loss.calculate(y_pred, y_batch)
            reg_loss = model.get_regularization_loss()
            
            # Calculate accuracy
            if model.accuracy is not None:
                model.accuracy.calculate(y_pred, y_batch)
            
            # Backward pass
            model.backward(y_pred, y_batch)
            
            # Update parameters
            model.update_params()
            
            epoch_data_loss += data_loss
            epoch_reg_loss += reg_loss
        
        # Calculate epoch statistics
        n_batches = len(train_loader)
        avg_data_loss = epoch_data_loss / n_batches
        avg_reg_loss = epoch_reg_loss / n_batches
        avg_loss = avg_data_loss + avg_reg_loss
        avg_acc = (model.accuracy.calculate_accumulated()
                   if model.accuracy is not None else None)
        
        # Validation
        val_loss, val_acc = model.evaluate(X_val, y_val)
        
        # Save to history
        training_history['train_loss'].append(avg_loss)
        training_history['train_acc'].append(avg_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['epoch'].append(epoch)
        
        # Monitor and save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0
            
            # Save model with improved validation accuracy
            model.save(str(MODEL_SAVE_PATH))
            improvement_marker = " [SAVED]"
        else:
            early_stop_counter += 1
            improvement_marker = ""
        
        # Print progress
        if epoch % 10 == 0:
            lr = (model.optimizer.current_lr
                  if hasattr(model.optimizer, 'current_lr') else LEARNING_RATE)
            print(f"Epoch {epoch:3d} | "
                  f"Loss: {avg_loss:.5f} | Acc: {avg_acc:.4f} | "
                  f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {lr:.6f}{improvement_marker}")
        
        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {early_stop_patience} epochs)")
            break
    
    elapsed_time = time.time() - start_time
    
    return {
        'history': training_history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'total_time': elapsed_time
    }


def evaluate_on_test(model: Model, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Evaluate trained model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    test_loss, test_acc = model.evaluate(X_test, y_test)
    return test_loss, test_acc


def print_training_summary(results: dict, test_loss: float, test_acc: float):
    """
    Print formatted training summary.
    
    Args:
        results: Training results dictionary
        test_loss: Loss on test set
        test_acc: Accuracy on test set
    """
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total Epochs: {results['total_epochs']}")
    print(f"Training Time: {results['total_time']:.2f} seconds ({results['total_time']/60:.2f} minutes)")
    print(f"\nBest Model (Epoch {results['best_epoch']}):")
    print(f"  Best Validation Accuracy: {results['best_val_acc']:.5f}")
    print(f"  Final Training Loss: {results['history']['train_loss'][-1]:.5f}")
    print(f"  Final Training Accuracy: {results['history']['train_acc'][-1]:.5f}")
    print(f"\nTest Set Performance:")
    print(f"  Test Loss: {test_loss:.5f}")
    print(f"  Test Accuracy: {test_acc:.5f}")
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("="*70)


# ==================== Main ====================

def main():
    """Main training pipeline."""
    try:
        # Set random seed for reproducibility
        np.random.seed(RANDOM_SEED)
        
        # Create output directory if needed
        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare dataset
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()
        
        # Build model
        print("\nBuilding model architecture...")
        model = build_model()
        
        # Train model
        results = train_model(model, X_train, y_train, X_val, y_val)
        
        # Reload best model and evaluate on test set
        print("\nLoading best model and evaluating on test set...")
        model = Model.load(str(MODEL_SAVE_PATH))
        test_loss, test_acc = evaluate_on_test(model, X_test, y_test)
        
        # Print summary
        print_training_summary(results, test_loss, test_acc)
        
        return model, results
        
    except Exception as e:
        print(f"Error during training: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    model, results = main()
