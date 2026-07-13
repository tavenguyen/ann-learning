# Fashion-MNIST Classifier

A production-quality Artificial Neural Network classifier for Fashion-MNIST using a custom neural network framework.

## Overview

This project implements a fully-connected neural network trained on the Fashion-MNIST dataset (10 classes, 28×28 grayscale images) using only the custom framework (no TensorFlow, Keras, or PyTorch).

### Performance Target
- **Validation Accuracy**: ~90%
- **Test Accuracy**: ~88%

## Project Structure

```
Model/MNIST Fashion/
├── fashion_mnist_train.py          # Training pipeline
├── fashion_mnist_recognition.py    # Inference & visualization
├── fashion_mnist.params            # Saved model (generated after training)
└── Fashion_Mnist_Images/
    └── test/                        # Test images for recognition
```

## File Documentation

### 1. fashion_mnist_train.py

**Purpose**: Train the Fashion-MNIST classifier with validation monitoring and best model saving.

**Key Features**:
- Automatic dataset loading from local directories
- Intelligent data preprocessing (normalization, flattening)
- Train/validation split (80/20)
- Label Smoothing ($\alpha=0.1$) to prevent over-confidence and improve generalization
- Mini-batch gradient descent training
- Real-time validation monitoring
- Best model checkpoint saving
- Early stopping after 15 epochs without improvement
- Comprehensive training logs and final report

**Network Architecture**:
```
Input (784 features)
    ↓
Dense(256, init="he") → LeakyReLU → Dropout(30%)
    ↓
Dense(128, init="he") → LeakyReLU → Dropout(30%)
    ↓
Dense(10) → Softmax
Output (10 classes)
```

**Design Rationale**:
- **Simplified Architecture**: 784→256→128→10 maintains high capacity while reducing training time and risk of overfitting.
- **LeakyReLU Activations**: Prevents "dying ReLU" problem by allowing a small gradient for negative values.
- **He Initialization**: Optimizes weight variance for LeakyReLU, ensuring stable signal flow through deep layers.
- **Label Smoothing ($\alpha=0.1$)**: Converts hard targets to soft distributions, reducing model over-confidence.
- **Dropout (30%)**: Prevents co-adaptation of neurons, improving generalization.
- **L2 Regularization (1e-5)**: Provides a light penalty on weights to prevent explosion without suppressing learning.
- **Adam Optimizer**: Adaptive learning rates for stable convergence.
- **Mini-batches (128)**: Balance between gradient noise and computational efficiency.

**Usage**:
```bash
cd Model/MNIST\ Fashion
python fashion_mnist_train.py
```

**Output**:
```
✓ Loaded dataset: train=(60000, 28, 28), test=(10000, 28, 28)
✓ Dataset prepared:
  Train: (48000, 784), (48000, 10)
  Validation: (12000, 784), (12000, 10)
  Test: (10000, 784), (10000,)

✓ Building model architecture...

======================================================================
TRAINING FASHION-MNIST CLASSIFIER
======================================================================
Epochs: 100 | Batch size: 128
Learning rate: 0.001 | L2 regularization: 1e-5
Dropout: 30% | Architecture: 784->256->128->10
======================================================================
...
```

**Key Parameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 128 | Good balance between stability and memory usage |
| Learning Rate | 0.001 | Standard for Adam optimizer |
| Learning Rate Decay | 1e-5 | Gentle decay for stable convergence |
| L2 Regularization | 1e-5 | Light penalty to prevent overfitting |
| Label Smoothing | $\alpha=0.1$ | Regularizes output distributions |
| Dropout Rates | 30% | Empirically optimal for this architecture |
| Early Stopping Patience | 15 | Prevents unnecessary training after plateau |
| Validation Split | 20% | Standard for monitoring |

---

### 2. fashion_mnist_recognition.py

**Purpose**: Load trained model and perform inference on test images with visualization.

**Key Features**:
- Model loading from saved parameters
- Automatic test image discovery
- Preprocessing pipeline (identical to training)
- Single and batch inference
- Matplotlib visualization with predictions
- Formatted console output
- Robustness to missing/invalid images
- Support for multiple image formats (JPG, PNG, BMP, GIF, TIFF)

**Usage**:
```bash
cd Model/MNIST Fashion
python fashion_mnist_recognition.py
```

**Output Example**:
```
✓ Loaded model from Model/MNIST Fashion/fashion_mnist.params

✓ Found 100 test image(s)

======================================================================
PREDICTIONS
======================================================================
  📄 img_001.jpg                    → [7] Sneaker          ( 95.2%)
  📄 img_002.png                    → [0] T-shirt/top      ( 87.3%)
  📄 img_003.jpg                    → [3] Dress            ( 92.1%)
...
======================================================================

======================================================================
STATISTICS
======================================================================
Total images processed: 100

Average confidence: 0.8956
Min confidence: 0.6234
Max confidence: 0.9987

Prediction distribution:
  [0] T-shirt/top    : 10 (10.0%)
  [1] Trouser        :  8 ( 8.0%)
  [2] Pullover       :  9 ( 9.0%)
  [3] Dress          :  7 ( 7.0%)
  [4] Coat           :  8 ( 8.0%)
  [5] Sandal         : 11 (11.0%)
  [6] Shirt          :  9 ( 9.0%)
  [7] Sneaker        : 12 (12.0%)
  [8] Bag            : 10 (10.0%)
  [9] Ankle boot     : 16 (16.0%)
======================================================================

✓ Recognition pipeline completed successfully!
```

**Class Labels**:
```
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
```

---

## Framework API Usage

### Components Used

**Layers**:
- `DenseLayer(n_inputs, n_neurons, weight_regularizer=None, init_type="he")` - Fully connected layer with optional He initialization

**Activations**:
- `Activation_LeakyReLU(alpha=0.01)` - Leaky Rectified Linear Unit
- `Activation_Softmax_Loss_CategoricalCrossEntropy()` - Combined loss + activation (supports soft labels)

**Regularization**:
- `Regularization_L2(strength=1e-5)` - L2 weight regularization

**Dropout**:
- `Layer_Dropout(dropout_rate=0.3)` - Dropout for regularization

**Optimizers**:
- `Optimizer_Adam(learning_rate=0.001, decay_rate=1e-5)` - Adam optimizer with decay

**Accuracy Metrics**:
- `Accuracy_CategoricalClassification()` - Multi-class accuracy (compatible with soft labels)

**Data Loading**:
- `DataLoader(X, y, batch_size=32, shuffle=True)` - Mini-batch iterator

**Model**:
- `Model()` - Neural network container
  - `.add(layer)` - Add layer
  - `.set(loss, optimizer, accuracy)` - Configure model
  - `.finalize()` - Prepare for training
  - `.forward(X, training=True)` - Inference
  - `.backward(y_pred, y_true)` - Backpropagation
  - `.update_params()` - Parameter updates
  - `.train(X, y, epochs, batch_size, validation_data)` - Training loop
  - `.evaluate(X, y)` - Validation/testing
  - `.predict(X)` - Inference
  - `.save(path)` - Save model
  - `.load(path)` - Load model

---

## Data Preprocessing Pipeline

### Training Preprocessing
```python
1. Load 28×28 uint8 images from Fashion-MNIST
2. Normalize: image / 255.0 → [0, 1]
3. Convert dtype: uint8 → float32
4. Flatten: (28, 28) → (784,)
5. Label Smoothing: Hard labels → Soft distributions (alpha=0.1)
Result: (N, 784) feature matrix and (N, 10) soft labels
```

### Inference Preprocessing
```
MUST be IDENTICAL to training:
1. Load image file (supports JPG, PNG, BMP, GIF, TIFF)
2. Convert to grayscale if needed
3. Resize to 28×28 if needed
4. Normalize: pixel / 255.0 → [0, 1]
5. Convert dtype: uint8 → float32
6. Flatten: (28, 28) → (784,)
```

**Critical**: Identical preprocessing ensures valid predictions.

---

## Training Strategy

### Phase 1: Warm-up (Epochs 0-20)
- High learning rate (0.001)
- Rapid loss decrease
- Building initial feature detectors

### Phase 2: Main Training (Epochs 20-50)
- Stable learning rate with gentle decay
- Continued improvement in validation accuracy
- Fine-tuning of decision boundaries

### Phase 3: Fine-tuning (Epochs 50+)
- Very low learning rate (decayed to ~0.0009)
- Diminishing returns on validation accuracy
- May overfit training set

### Early Stopping
- Monitors validation accuracy
- Stops if no improvement for 15 consecutive epochs
- Prevents overfitting and saves computation

---

## Best Practices Demonstrated

### 1. Clean Architecture
- Separate functions for each responsibility
- Reusable preprocessing pipeline
- Clear separation of train/predict

### 2. Reproducibility
- Fixed random seed (42)
- Deterministic data splitting
- Saved model weights

### 3. Robustness
- Graceful error handling
- Informative error messages
- Directory existence checks
- Image format validation

### 4. Monitoring
- Real-time training logs
- Best model checkpoint
- Comprehensive final report
- Prediction statistics

### 5. Documentation
- Inline comments for design decisions
- Clear variable names
- Docstrings for all functions
- Type hints

---

## Requirements

### Python Packages
- `numpy` - Numerical computations
- `pillow` - Image loading
- `matplotlib` - Visualization

### Framework
- Custom ANN framework in `Model/model.py`

### Dataset
- Fashion-MNIST images provided in `Fashion_Mnist_Images/` directory

---

## Troubleshooting

### Issue: "No test images found"
**Solution**: Create `Model/MNIST Fashion/Fashion_Mnist_Images/test/` and add PNG/JPG files

### Issue: Low accuracy during inference
**Solution**: Verify preprocessing is identical to training (normalize to [0,1])

### Issue: Model file too large
**Solution**: Normal - Weights for the current architecture occupy several MBs.

---

## Performance Analysis

### Expected Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| Train Accuracy | ~97-98% | Some overfitting acceptable |
| Validation Accuracy | ~89-90% | Good generalization |
| Test Accuracy | ~88-89% | Realistic performance |
| Training Time | ~2-3 minutes | CPU-based |
| Model Size | ~2-5 MB | Parameter file |

### Optimization Opportunities
1. **Wider networks** (1024+ hidden units) → Slightly higher accuracy, more parameters
2. **Deeper networks** (6+ layers) → Marginal gains, risk of overtraining
3. **Batch normalization** → Faster convergence (not available in framework)
4. **Data augmentation** → Better generalization (requires preprocessing changes)

---

## Code Quality

✓ **PEP8 Compliant**
✓ **Production-Ready**
✓ **Comprehensive Documentation**
✓ **Robust Error Handling**
✓ **Efficient Memory Usage**
✓ **Clear Variable Naming**

---

## References

### Fashion-MNIST Paper
- [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747)

### Dataset Statistics
- 70,000 samples (60,000 train + 10,000 test)
- 10 classes (clothing and footwear)
- 28×28 grayscale images
- Balanced class distribution

---

## License

This implementation uses the custom neural network framework provided in the project.

---

## Author

Implemented as a demonstration of best practices in deep learning engineering using a custom framework.

