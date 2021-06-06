import numpy as np

import random

class NeuralNet:

    # Define the structure of our neural net - the size of all the layers, initial w's and b's etc...
    def __init__(self, layer_dims, for_digits=False):
        self.for_digits = for_digits
        
        self.__generate_layers(layer_dims)

    def __generate_layers(self, layer_dims):
        self.n_layers = len(layer_dims)
        self.layer_dims = layer_dims

        # Use a Gaussian distribution to assign initial values to w & b
        self.w = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(layer_dims[:-1], layer_dims[1:])]
        self.b = [np.zeros((y, 1)) for y in layer_dims[1:]]
    
    def __resize_input_layer(self, size):
        self.layer_dims[0] = size
        self.__generate_layers(self.layer_dims)
        
    # This defines our sigmoid neuron output function.
    def sigmoid(self, weighted_input):
        return 1 / (1 + np.exp(-weighted_input))

    # Helper function for backprop - derivative of sigmoid
    # This is quite a simple & nice derivative since we use an exponential in sigmoid
    def del_sigmoid(self, weighted_input):
        return self.sigmoid(weighted_input) * (1 - self.sigmoid(weighted_input))

    def get_output(self, inp):
        # Feed the inputs forward through the network.
        # 'inp' gets re-assigned at each step - you can think of this as updating the inputs for each
        # neuron layer.
        for w, b in zip(self.w, self.b):
            inp = self.sigmoid(np.dot(w, inp) + b)

        # After the final pass our 'inputs' are the outputs of the final neural layer - the output.
        return inp

    def del_cost(self, out_activations, out):
        # Change in cost - this is simply the difference in the activation ouput and the actual output of a neuron
        # Since we are using cross-entropy
        return out_activations - out

    def backprop(self, inp, out):

        # Assign shape
        del_w = [np.zeros(w.shape) for w in self.w]
        del_b = [np.zeros(b.shape) for b in self.b]

        # Feed forward the inputs for each layer to get the output layer

        # Activations stores the output of each node on a layer-by-layer bais
        activation = inp
        activations = [inp]
        weighted_inps = []
        for w, b in zip(self.w, self.b):
            weighted_inp = np.dot(w, activation) + b
            weighted_inps.append(weighted_inp)
            activation = self.sigmoid(weighted_inp)
            activations.append(activation)

        # Calculate the error in the output layer
        # Note * with numpy is Hadamard product - elementwise multiplication
        error = self.del_cost(activations[-1], out) * self.del_sigmoid(weighted_inps[-1])

        # For the following, please refer to the equations for backprop.
        # Note del cost / del weight = activation^(l-1) . error
        del_w[-1] = np.dot(error, activations[-2].transpose())
        # Note del cost / del bias = the error function
        del_b[-1] = error

        # Begin backprop - almost the same process as final layer, except the error is calculated slightly differently
        # Of course, start at index 2 to n_layers - we skip the final output layer 
        # (we use the -ve value of this index as we are propagating from the output layer backwards
        #       made easier with python's -ve list indexing)
        for layer_index in range(2, self.n_layers):
            # Again, refer to backprop equations for this.
            del_sigma = self.del_sigmoid(weighted_inps[-layer_index])
            error = np.dot(self.w[-layer_index + 1].transpose(), error) * del_sigma
            del_w[-layer_index] = np.dot(error, activations[-layer_index - 1].transpose())
            del_b[-layer_index] = error

        return del_w, del_b

    def compute_batch_gradient(self, batch):
        # Fill arrays so we have a structure of the correct shape to work with
        del_w = [np.zeros(w.shape) for w in self.w]
        del_b = [np.zeros(b.shape) for b in self.b]

        # For each training sample
        for data, label in batch:
            # Do backprop - update the partial derivatives w.r.t. the weights and biases
            # Note we do not set the weights and biases, this is done after each batch
            delta_del_w, delta_del_b = self.backprop(data, label)
            del_w = [dw + delta_dw for dw, delta_dw in zip(del_w, delta_del_w)]
            del_b = [db + delta_db for db, delta_db in zip(del_b, delta_del_b)]

        return del_w, del_b

    def grad_descent(self, training_data, testing_data, learning_rate, batch_size, epochs, do_tests=False):

        # Cache the data size so we know how many batches to generate
        data_size = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            # Create equally sized batches based on param passed in.
            batches = [training_data[j:j+batch_size] for j in range(0, data_size, batch_size)]
            for batch in batches:
                # For each batch adjust the weights proportionally to the gradient w.r.t. the batch
                del_w, del_b = self.compute_batch_gradient(batch)
                self.w = [w - ((learning_rate / len(batch)) * dw) for w, dw in zip(self.w, del_w)]
                self.b = [b - ((learning_rate / len(batch)) * db) for b, db in zip(self.b, del_b)]

            if do_tests and not (testing_data is None): 
                successes = self.test_current_params(testing_data)
                print(f"E={i}: {successes}/{len(testing_data)}, {successes*100/len(testing_data)}%")

    def train(self, training_data, training_labels,
              testing_data=None, testing_labels=None,
              batch_size=20, epochs=35,
              learning_rate = 0.5,
              data_zipped=False, auto_resize=True):
        
        # Accommodate for testing interface
        if self.for_digits:
            
            if auto_resize and (not training_data.shape[1] == self.layer_dims[0]):
                self.__resize_input_layer(training_data.shape[1])
            
            training_data, training_labels = self.format_digit_data(training_data, training_labels)
            
            if not (testing_data is None or testing_labels is None):
                testing_data, testing_labels = self.format_digit_data(testing_data, testing_labels)

        # Put the data in an easy arrangement, arrays of tuples (inputs, expected output)
        if not data_zipped:
            zipped_data = [(data, label) for data, label in zip(training_data, training_labels)]
            zipped_test = [(data, label) for data, label in zip(testing_data, testing_labels)] if not (testing_data is None or testing_labels is None) else None
        else:
            zipped_data = training_data
            zipped_test = testing_data if not (testing_data is None or testing_labels is None) else None

        self.grad_descent(zipped_data, zipped_test, learning_rate, batch_size, epochs)

    def test_current_params(self, testing_data):

        if not self.for_digits:
            # For spam filter round confidences above 0.5 to 1 and those below to 0 and compare to the label.
            results = [(1 if self.get_output(inp) > 0.5 else 0, expected[0][0]) for inp, expected in testing_data]
            # Sum them all up - possible time complexity improvement by doing this above?
            return sum(1 if out == expected else 0 for out, expected in results)
        else:
            # For digit recognition, we compare the *arg* of the maximum entry in the column vector as these correspond
            # To digits 0 - 9
            results = [(np.argmax(self.get_output(inp)), np.argmax(expected)) for inp, expected in testing_data]
            return sum(1 if out == expected else 0 for out, expected in results)
        
    def predict(self, test_data):
        if self.for_digits:
            # Reshape to fit neural net's format
            td = [np.reshape(x, (x.shape[0], 1)) for x in test_data]
            return [np.argmax(self.get_output(d)) for d in td]
        else:
            td = [np.reshape(x, (54, 1)) for x in test_data]
            return [1 if self.get_output(d) > 0.5 else 0 for d in td]
        
        
    # Data formatting helper funcs - moved here so can be easily called by
    # Methods so that this class can pass the automated tests.
    def format_label(self, label):
        # We need to turn the digits into a column vector of all zeroes except the number - a 1
        lb = np.zeros((10, 1))
        lb[label] = 1
        return lb

    def format_digit_data(self, inps, labels, shape=None):
        if shape is None:
            shape = inps.shape[1]
        
        # Format the data into column vectors
        lb = [self.format_label(l) for l in labels]

        # Flatten the images into a column vector
        ipt = [np.reshape(i, (shape, 1)) for i in inps]

        return ipt, lb
