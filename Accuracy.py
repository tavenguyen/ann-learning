import numpy as np
class Accuracy:
    def calculate(self, y_pred, y_true):
        comparisions = (y_pred == y_true)
        accuracy = np.mean(comparisions)
        return accuracy