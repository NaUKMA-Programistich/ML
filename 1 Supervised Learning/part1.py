import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class MyLinearRegression:
    def __init__(self, weights_init='random', add_bias = True, learning_rate=1e-5, 
        num_iterations=1_000, verbose=False, max_error=1e-5):
        ''' Linear regression model using gradient descent 

        # Arguments
            weights_init: str
                weights initialization option ['random', 'zeros']
            add_bias: bool
                whether to add bias term 
            learning_rate: float
                learning rate value for gradient descent
            num_iterations: int 
                maximum number of iterations in gradient descent
            verbose: bool
                enabling verbose output
            max_error: float
                error tolerance term, after reaching which we stop gradient descent iterations
        '''

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights_init = weights_init
        self.add_bias = add_bias
        self.verbose = verbose
        self.max_error = max_error
    
    def initialize_weights(self, n_features):
        ''' weights initialization function '''
        if self.weights_init == 'random':
            ################

            # YOUR CODE HERE
            weights = # TODO

            ################
        elif self.weights_init == 'zeros':
            ################

            # YOUR CODE HERE
            weights = # TODO

            ################
        else:
            raise NotImplementedError
        return weights

    def cost(self, target, pred):
        ''' calculate cost function 
        
            # Arguments:
                target: np.array
                    array of target floating point numbers 
                pred: np.array
                    array of predicted floating points numbers
        '''
        ################

        # YOUR CODE HERE
        loss = None

        ################
        return loss

    def fit(self, x, y):
        if self.add_bias:
            ################

            # YOUR CODE HERE

            ################

        self.weights = self.initialize_weights(x.shape[1])

        for i in range(self.num_iterations):
            ################

            # YOUR CODE HERE
            # step 1: calculate current_loss value

            # step 2: calculate gradient value

            # step 3: update weights using learning rate and gradient value

            # step 4: calculate new_loss value

            # step 5: if new_loss and current_loss difference is greater than max_error -> break;
            #         if iteration is greater than max_iterations -> break
        
            ################
    
    def predict(self, x):
        ''' prediction function '''
        ################

        # YOUR CODE HERE
        y_hat = # TODO

        ################
        return y_hat



def normal_equation(X, y):
    ''' TODO: implement normal equation '''
    return None



if __name__ == "__main__":
    # generating data samples
    x = np.linspace(-5.0, 5.0, 100)[:, np.newaxis]
    y = 29 * x + 40 * np.random.rand(100,1)

    # normalization of input data
    x /= np.max(x)

    plt.title('Data samples')
    plt.scatter(x, y)
    plt.savefig('data_samples.png')


    # Sklearn linear regression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    y_hat_sklearn = sklearn_model.predict(x)

    plt.title('Data samples with sklearn model')
    plt.scatter(x, y)
    plt.plot(x, y_hat_sklearn, color='r')
    plt.savefig('sklearn_model.png')
    print('Sklearn MSE: ', mean_squared_error(y, y_hat_sklearn))

    # Your linear regression model
    my_model = MyLinearRegression()
    my_model.fit(x, y)
    y_hat = my_model.predict(x)

    plt.title('Data samples with my model')
    plt.scatter(x, y)
    plt.plot(x, y_hat, color='r')
    plt.savefig('my_model.png')
    print('My MSE: ', mean_squared_error(y, y_hat))

    # Normal equation
    weights = normal_equation(x, y)
    y_hat_normal = x @ weights

    plt.title('Data samples with normal equation')
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal, color='r')
    plt.savefig('normal_equation.png')
    print('Normal equation MSE: ', mean_squared_error(y, y_hat_normal))