import random

# input: vector w (trong so), x (inputs)
def compute_output(w, x):
    z = 0
    for i in range(len(x)):
        z += w[i] * x[i]
    
    # apply sign function
    if z < 0:
        return -1
    
    return 1

def showing_learning(weights):
    print(weights)
            
random.seed(10)
x_train = [(1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0)]
y_true = [1.0, 1.0, 1.0, -1.0]
learning_rate = 0.1
weights = [random.random() for i in range(len(x_train[0]))] # List Comprehension
print(weights)

all_corrects = False
while not all_corrects:
    all_corrects = True
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_true[i]

        z = compute_output(weights, x)
        if(z != y):
            for j in range(len(weights)):
                weights[j] += (y * learning_rate * x[j])

            all_corrects = False
            showing_learning(weights)
    