import numpy as np
import matplotlib.pyplot as plt

#utils
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return (A, Z)

def relu(Z):
    A = np.copy(Z)
    A[A<0] = 0
    return (A, Z)

def sigmoid_backward(dA, Z):
    sig, cache = sigmoid(Z)
    grad_sig = sig*(1-sig)
    dZ = dA*grad_sig
    return dZ

def relu_backward(dA, Z):
    Z[Z>=0] = 1
    Z[Z<0] = 0
    dZ = dA*Z
    return dZ


class Basic_NN(object):
    def __init__(self, train_x, train_y, layers_dims, learning_rate=0.001):
        self.__layers_dims = layers_dims
        self.__L = len(layers_dims)-1
        self.train_x = train_x
        self.train_y = train_y
        self.__m = train_x.shape[1]
        self.parameters = self.__initialize_parameters()
        self.learning_rate = learning_rate

    def __initialize_parameters(self):
        parameters = {}
        for l in range(self.__L):
            parameters['W'+str(l+1)] = np.random.randn(self.__layers_dims[l+1], self.__layers_dims[l])*0.01
            parameters['b'+str(l+1)] = np.zeros((self.__layers_dims[l+1], 1))
        return parameters

    def __linear_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        return Z, cache

    def __linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self.__linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        else:
            A, activation_cache = relu(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

    def __L_model_forward(self, parameters, x_data=np.array([])):
        caches = []
        if x_data.any():
            # add dim check exception handeling
            A = x_data
        else:
            A = self.train_x
        for l in range(self.__L-1):
            A_prev = A
            W = parameters['W'+str(l+1)]
            b = parameters['b'+str(l+1)]
            A, cache = self.__linear_activation_forward(A_prev, W, b, "relu")
            caches.append(cache)
        W = parameters['W' + str(self.__L)]
        b = parameters['b' + str(self.__L)]
        AL, cache = self.__linear_activation_forward(A, W, b, "sigmoid")
        caches.append(cache)
        return AL, caches

    def __compute_cost(self, AL, tr_y=np.array([])):
        if tr_y.any():
            m = tr_y.shape[1]
            l1 = np.log(AL)
            l2 = np.log(1 - AL)
            cost = np.sum(tr_y * l1 + (1 - tr_y) * l2) / (-m)
            return cost
        else:
            l1 = np.log(AL)
            l2 = np.log(1 - AL)
            cost = np.sum(self.train_y * l1 + (1 - self.train_y) * l2) / (-self.__m)
            return cost

    def __linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        dW = np.dot(dZ, A_prev.T) / self.__m
        db = np.sum(dZ, axis=1, keepdims=True) / self.__m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def __linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
        else:
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def __L_model_backward(self, AL, caches):
        grads = {}
        L = self.__L
        m = self.__m
        Y = self.train_y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.__linear_activation_backward(dAL,
                                                                                                          current_cache,
                                                                                                          "sigmoid")
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.__linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads

    def __update_parameters(self, parameters, grads):
        L = self.__L
        for l in range(L):
            parameters["W" + str(l + 1)] -= grads["dW" + str(l + 1)] * self.learning_rate
            parameters["b" + str(l + 1)] -= grads["db" + str(l + 1)] * self.learning_rate
        return parameters

    def Train(self, epochs=100, print_cost=False):
        np.random.seed(1)
        tr_costs = []  # keep track of cost
        vl_costs = []
        parameters = self.parameters
        # Loop (gradient descent)
        for i in range(epochs):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.__L_model_forward(parameters)
            # Compute cost.
            cost = self.__compute_cost(AL)
            # Backward propagation.
            grads = self.__L_model_backward(AL, caches)
            # Update parameters.
            parameters = self.__update_parameters(parameters, grads)
            # Print the cost every 100 training example
            if i % 10 == 0:
                if print_cost:
                    print("Cost after iteration %i: %f" % (i, cost))
                tr_costs.append(cost)
        # plot the cost
        plt.plot(np.squeeze(tr_costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
        self.parameters = parameters
        return parameters

    def predict(self, x_data):
        out, _ = self.__L_model_forward(self.parameters, x_data)
        return out

    def evaluate(self, x_data, y_data):
        out = self.predict(x_data)
        cost = self.__compute_cost(out, y_data)
        return cost