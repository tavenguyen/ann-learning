# Artificial Neural Network Learning Repository

This is my learning journey through building neural networks from scratch. The repository contains implementations of fundamental concepts like perceptron, gradient descent, backpropagation, and various optimization techniques. I've also built a custom neural network library to understand how everything works under the hood.

## Repository Structure

The project is organized into several folders, each focusing on different aspects of neural networks:

- **Perceptron/**: Basic perceptron implementation and examples
- **Gradient_Descent/**: Gradient descent fundamentals and linear/quadratic regression examples
- **Backpropagation/**: Implementation of backpropagation algorithm, dense layers, and activation functions
- **Optimization/**: Different optimization algorithms including SGD, momentum, learning rate decay, and regularization
- **Model/**: The custom neural network library with complete implementation
  - `model.py`: Main implementation containing layers, activations, loss functions, data generators
  - `docs/`: Detailed documentation for each component
- **Points/**: Data generation utilities
- **documents/**: Theoretical notes and mathematical explanations

## Quick Start / Usage Examples

Here's a simple example of building and training a neural network for classification on spiral data:

```python
from Model.model import Dense, Activation_ReLU, SpiralData, Layer_DropOut
from Model.model import Activation_Softmax_Loss_CategoricalCrossEntropy
from Model.model import Accuracy_CategoricalClassification

# Create training data
n_classes = 3
n_points = 400
training_data = SpiralData(n_points=n_points, n_classes=n_classes)
X = training_data.P
Y = training_data.L

# Build the network
layer1 = Dense(n_inputs=2, n_neurons=32)
activation1 = Activation_ReLU()
dropout_layer1 = Layer_DropOut(dropout_rate=0.05)

layer2 = Dense(n_inputs=32, n_neurons=32)
activation2 = Activation_ReLU()
dropout_layer2 = Layer_DropOut(dropout_rate=0.05)

layer_output = Dense(n_inputs=32, n_neurons=3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
accuracy_function = Accuracy_CategoricalClassification()

# Forward pass
layer1.forward(X)
activation1.forward(layer1.output)
dropout_layer1.forward(activation1.output)

layer2.forward(dropout_layer1.output)
activation2.forward(layer2.output)
dropout_layer2.forward(activation2.output)

layer_output.forward(dropout_layer2.output)
loss = loss_activation.forward(layer_output.output, Y)
```

For a quadratic regression example, you can check `Model/Quadratic_Regression_Model.py`.

## Documentation Links

Each component has detailed documentation in the `Model/docs/` folder:

- **DataPoints.md**: How to use different data generators (Line, Quadratic, SpiralData)
- **Activations.md**: Available activation functions and how they work
- **Losses.md**: Loss functions for different problem types
- **Layers.md**: Dense layers and other layer types
- **Optimizers.md**: Complete guide to different optimizers
- **Accuracy.md**: Accuracy metrics for classification
- **EX1.md**: Example 1 - Classification with neural network
- **EX2.md**: Example 2 - Another classification example

The `documents/` folder contains more detailed mathematical explanations and derivations.

## Installation & Requirements

You'll need Python 3.7 or higher with NumPy. To set up:

```bash
pip install numpy
```

If you want to visualize results, also install matplotlib:

```bash
pip install matplotlib
```

That's it. All the code uses only these libraries, no deep learning frameworks required.

## References & Resources

This learning journey was guided by:

- 3Blue1Brown's Neural Networks video series on YouTube
- "Deep Learning" book by Goodfellow, Bengio, and Courville
- Various online courses and tutorials on machine learning fundamentals

The `documents/` folder in this repository also contains my notes on different concepts like backpropagation, accuracy metrics, and model serialization.
