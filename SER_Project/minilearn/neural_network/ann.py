import numpy as np

class MLPClassifier:
    def __init__(self, hidden_layer_sizes=100, learning_rate_init=0.01, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

        self.classes = None
        self.n_classes = None
        self.weights_input_hidden = None
        self.weights_hidden_output = None
        self.bias_hidden = None
        self.bias_output = None


    # hidden activation function to zero out negative weighted sums
    def _relu(self, z):
        return np.maximum(0, z)


    # used in backpropegation to tell the gradient if the neuron contributed
    def _relu_derivative(self, activation):
        return (activation > 0).astype(float)


    # normalizes raw output values to add to 1 across classes, giving probabilities
    def _softmax(self, z):
        # shifting avoids overflow
        shifted = z - z.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)

        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


    # makes a binary matrix which identifies the class of each row
    def _binary_matrix(self, y):
        # maps each label to its index in the sorted class array
        class_indices = np.searchsorted(self.classes, y)

        binary_matrix = np.zeros((y.size, self.n_classes))
        binary_matrix[np.arange(y.size), class_indices] = 1

        return binary_matrix

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        # basically, the answer key
        targets = self._binary_matrix(y)

        # weights are random so that neurons learn different things
        self.weights_input_hidden = np.random.randn(n_features, self.hidden_layer_sizes) * np.sqrt(2 / n_features)
        self.weights_hidden_output = np.random.randn(self.hidden_layer_sizes, self.n_classes) * np.sqrt(2 / self.hidden_layer_sizes)

        self.bias_hidden = np.zeros((1, self.hidden_layer_sizes))
        self.bias_output = np.zeros((1, self.n_classes))

        for iter in range(self.max_iter):
            # forward propegation

            #compute weighted sum
            hidden_activation = self._relu(X @ self.weights_input_hidden + self.bias_hidden)

            # raw unnormalized output values
            output_logits = hidden_activation @ self.weights_hidden_output + self.bias_output

            # normalized output values
            output_probs = self._softmax(output_logits)

            # back propegation

            # computes how much each weight contributed to the error
            output_gradient = (output_probs - targets) / n_samples

            # propegates error back through the output weights
            # has to be transpose to match dimensions
            # also, zeros out neurons which did not contribute
            hidden_gradient = output_gradient @ self.weights_hidden_output.T * self._relu_derivative(hidden_activation)


            # update the weights to reduce the loss
            self.weights_input_hidden -= self.learning_rate_init * X.T @ hidden_gradient
            self.weights_hidden_output -= self.learning_rate_init * hidden_activation.T @ output_gradient
            
            self.bias_output -= self.learning_rate_init * output_gradient.sum(axis=0)
            self.bias_hidden -= self.learning_rate_init * hidden_gradient.sum(axis=0)

        return self


    # returns the probabilities of a row being in each class
    def predict_proba(self, X):
        hidden_activation = self._relu(X @ self.weights_input_hidden + self.bias_hidden)

        return self._softmax(hidden_activation @ self.weights_hidden_output + self.bias_output)


    # returns the predicted class using the highest value from class probabilities
    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]