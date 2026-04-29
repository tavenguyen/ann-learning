import numpy as np
import random

def compute_output(x, weights): 
    return np.dot(x, weights)

random.seed(10)
# 3x4
inputsLayer1 = np.array([[1, 2, 3, 4], 
                         [5, 6, 7, 8], 
                         [9, 10, 11, 12]])

# 3x4
weights_list = []

for i in range(len(inputsLayer1)):
    row = []
    for j in range(len(inputsLayer1[0])):
        row.append(random.random())
    weights_list.append(row)

weightsLayer1 = np.array(weights_list)

bias1 = [random.random() for i in range(len(inputsLayer1))]

weightsLayer2 = np.random.rand(len(inputsLayer1), len(inputsLayer1))

bias2 = np.array([0.1, 0.2, 0.3])

outputLayer1 = compute_output(inputsLayer1, weightsLayer1.T) + bias1
outputLayer2 = compute_output(outputLayer1, weightsLayer2.T) + bias2

print(outputLayer1)
print(outputLayer2)