import numpy as np
from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.layers = [ConvolutionalLayer(in_channels=input_shape[2], out_channels=input_shape[2],
                                          filter_size=conv1_channels, padding=2),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, stride=2),
                       ConvolutionalLayer(in_channels=input_shape[2], out_channels=input_shape[2],
                                          filter_size=conv2_channels, padding=2),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, stride=2),
                       Flattener(),
                       FullyConnectedLayer(n_input=192, n_output=n_output_classes)]

    def zero_grad(self):
        for param in self.params().values():
            param.grad = 0

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

    # TODO Compute loss and fill param gradients wo/ l2
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        self.zero_grad()
        output_from_layer_forward = self.forward_pass(X)
        loss, grad = softmax_with_cross_entropy(output_from_layer_forward, y)
        output_from_layer_backward = self.backward_pass(grad)

        return loss

    def predict(self, X):
        preds = self.forward_pass(X)
        y_pred = np.argmax(preds, axis=-1)

        return y_pred

    def params(self):
        # TODO: Aggregate all the params from all the layers which have parameters
        result = {"cl1_W": self.layers[0].params()['W'],
                  "cl1_B": self.layers[0].params()['B'],
                  "cl2_W": self.layers[3].params()['W'],
                  "cl2_B": self.layers[3].params()['B'],
                  "fcl1_W": self.layers[7].params()['W'],
                  "fcl1_B": self.layers[7].params()['B']}
        return result
