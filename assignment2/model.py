import numpy as np
from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        # TODO Create necessary layers
        self.layers = [FullyConnectedLayer(n_input, hidden_layer_size),
                       ReLULayer(),
                       FullyConnectedLayer(hidden_layer_size, n_output)]
        self.reg = reg

    # TODO Set parameter gradient to zeros
    def zero_grad(self):
        for param in self.params().values():
            param.grad = 0

    # TODO Compute loss and fill param gradients
    #  by running forward and backward passes through the model
    def forward_pass(self, X):
        output_from_layer_forward = X
        for layer in self.layers:
            output_from_layer_forward = layer.forward(output_from_layer_forward)
        return output_from_layer_forward

    def backward_pass(self, grad):
        output_from_layer_backward = grad
        for layer in reversed(self.layers):
            output_from_layer_backward = layer.backward(output_from_layer_backward)
        return output_from_layer_backward

    def l2_reg(self):
        loss = 0
        for param in self.params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += reg_grad
        return loss

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        self.zero_grad()

        output_from_layer_forward = self.forward_pass(X)
        loss, grad = softmax_with_cross_entropy(output_from_layer_forward, y)
        output_from_layer_backward = self.backward_pass(grad)

        loss += self.l2_reg()
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        preds = self.forward_pass(X)
        y_pred = np.argmax(preds, axis=1)

        return y_pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {"fcl1_W": self.layers[0].params()['W'],
                  "fcl1_B": self.layers[0].params()['B'],
                  "fcl2_W": self.layers[2].params()['W'],
                  "fcl2_B": self.layers[2].params()['B'], }
        return result
