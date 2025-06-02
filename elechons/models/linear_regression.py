import numpy as np

class Prediction:
    input: np.ndarray
    prediction: np.ndarray

    def __init__(self, input, prediction):
        self.input = input
        self.prediction = prediction
    
    def residuals(self):
        return self.input - self.prediction
    
    def rmse(self):
        return np.sqrt(np.mean(np.pow(self.residuals(), 2)))
    
    def rmse_percent(self):
        return 100 * self.rmse() / np.sqrt(np.mean(np.pow(self.input, 2)))

# rows are features, cols are time domain samples
def linear_auto_regression(data):
    precision = np.linalg.inv(data @ data.T)
    autocov = data[:, :-1] @ data[:, 1:].T

    return Prediction(data[:, 1:], (data[:, :-1].T @ precision @ autocov).T)

def linear_exp_SGD(data, transition_learning_coefficient=0.1, div_by_zero_offset=1e-8, init_eye_coefficient=0.5): #, bias_learning_coefficient=0.01, div_by_zero_offset=1e-8):
    transition_matrix = init_eye_coefficient * np.eye(data.shape[0])
    transition_history = np.empty((data.shape[0], data.shape[0], data.shape[1]))
    transition_history[:, :, 0] = transition_matrix
    # bias = data[:, 0]
    predictions = np.empty((data.shape[0], data.shape[1] - 1))

    transition_sum = np.zeros_like(transition_matrix)
    transition_squared_sum = np.zeros_like(transition_matrix)

    # bias_sum = np.zeros_like(bias)
    # bias_squared_sum = np.zeros_like(bias)

    for i in range(data.shape[1] - 1):
        # predictions[:, i] = transition_matrix @ (data[:, i] - bias) + bias
        predictions[:, i] = transition_matrix @ data[:, i]
        error = data[:, i+1] - predictions[:, i]

        transition_learning_rate = transition_learning_coefficient / (np.sum(np.pow(data[:, i], 2)) + div_by_zero_offset)
        transition_matrix += transition_learning_rate * np.outer(error, data[:, i])
        transition_history[:, :, i + 1] = transition_matrix
        # bias += bias_learning_coefficient * (data[:, i+1] - bias)

        transition_sum += transition_matrix
        transition_squared_sum += np.pow(transition_matrix, 2)

        # bias_sum += bias
        # bias_squared_sum += np.pow(bias, 2)
    
    transition_mean = transition_sum / (data.shape[1] - 1)
    transition_var = transition_squared_sum / (data.shape[1] - 1) - np.pow(transition_mean, 2)

    # bias_mean = bias_sum / (data.shape[1] - 1)
    # bias_var = bias_squared_sum / (data.shape[1] - 1) - np.pow(bias_mean, 2)
    
    return Prediction(data[:, 1:], predictions), transition_mean, transition_var, transition_history #, bias_mean, bias_var


