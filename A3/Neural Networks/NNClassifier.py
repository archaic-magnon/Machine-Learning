
import pandas as pd
import numpy as np
import time


class NNClassifier:
    def __init__(self, mini_batch_size=100, n_feature=28 * 28, hidden_layers=[20, 30, 40, 50, 60, 70], n_target=26, eta=0.1, hidden_activation="sigmoid", adaptive=False, max_epoch=1500, random_state=1, n_epoch_no_change=10, weight_init="xavier", verbose=False):
        self.mini_batch_size = mini_batch_size

        self.n_feature = n_feature
        self.hidden_layers = hidden_layers
        self.n_target = n_target
        self.layers = [self.n_feature] + self.hidden_layers + [self.n_target]

        self.hidden_activation_name = hidden_activation
        self.verbose = verbose
        self.max_epoch = max_epoch
        self.n_epoch_no_change = n_epoch_no_change
        self.eta = eta
        self.adaptive = adaptive

        self.random_state_val = random_state
        self.random_state = np.random.RandomState(random_state)

        self.factor = 2. if hidden_activation == 'sigmoid' else 6
        self.weight_init = weight_init

        if self.weight_init == "uniform":
            self.uniformInitialization()
        elif self.weight_init == "he":
            self.heInitialization()
        else:
            self.xavierInitialization()

        self.activation = self.sigmoid
        self.activation_dash = self.sigmoid_dash

        self.hidden_activation = self.sigmoid if hidden_activation == "sigmoid" else self.relU if hidden_activation == "relu" else self.sigmoid
        self.hidden_activation_dash = self.sigmoid_dash if hidden_activation == "sigmoid" else self.relU_dash if hidden_activation == "relu" else self.sigmoid_dash

        self.output_activation = self.sigmoid
        self.output_activation_dash = self.sigmoid_dash

    def __str__(self):
        s = f"NNClassifier(mini_batch_size={self.mini_batch_size}, n_feature={self.n_feature}, hidden_layers={self.hidden_layers}, n_target={self.n_target}, eta={self.eta}, hidden_activation={self.hidden_activation_name}, adaptive={self.adaptive}, random_state={self.random_state_val})"
        return s

    def heInitialization(self):
        self.b = [0] + [self.random_state.uniform(-self.getHeBound(n_unit1, n_unit2), self.getHeBound(n_unit1, n_unit2), (n_unit2, 1)) for n_unit1, n_unit2 in zip(self.layers[:-1], self.layers[1:])]
        self.w = [0] + [self.random_state.uniform(-self.getHeBound(n_unit1, n_unit2), self.getHeBound(n_unit1, n_unit2), (n_unit2, n_unit1)) for n_unit1, n_unit2 in zip(self.layers[:-1], self.layers[1:])]

    def xavierInitialization(self):
        self.b = [0] + [self.random_state.uniform(-self.getXavierBound(n_unit1, n_unit2), self.getHeBound(n_unit1, n_unit2), (n_unit2, 1)) for n_unit1, n_unit2 in zip(self.layers[:-1], self.layers[1:])]
        self.w = [0] + [self.random_state.uniform(-self.getXavierBound(n_unit1, n_unit2), self.getHeBound(n_unit1, n_unit2), (n_unit2, n_unit1)) for n_unit1, n_unit2 in zip(self.layers[:-1], self.layers[1:])]

    def uniformInitialization(self):
        self.b = [0] + [self.random_state.uniform(-0.01, 0.01, (n_unit2, 1)) for n_unit1, n_unit2 in zip(self.layers[:-1], self.layers[1:])]
        self.w = [0] + [self.random_state.uniform(-0.01, 0.01, (n_unit2, n_unit1)) for n_unit1, n_unit2 in zip(self.layers[:-1], self.layers[1:])]

    def getHeBound(self, unit1, unit2):
        # return 0.01
        return np.sqrt(self.factor / (unit1))

    def getXavierBound(self, unit1, unit2):
        return np.sqrt(self.factor / (unit1 + unit2))

    def converge(self, j1, j2, epsilon=1e-4):
        if self.adaptive:
            epsilon = 1e-12
        if (abs(j1 - j2) < epsilon):
            return True
        else:
            return False

    def cost(self, y, y_hat):
        '''
        Input:
            y: original label, s*o (s = no. of example, o = no. of output unit)
            y_hat: s*o
        '''
        m = y.shape[0]
        j = (1. / (2 * m)) * np.sum((y - y_hat)**2)
        return j

    def shuffle_data(self, x, y):
        p = self.random_state.permutation(len(x))
        return x[p], y[p]

    def getEncodedY(self, y):
        '''
        Input:
            y: s*1 (s = no. of example)

        Output:
            y = s*n_target (n_target = no. of target class)
        '''
        y_h = np.zeros((y.shape[0], self.n_target))
        for i, item in enumerate(y):
            y_h[i][item[0]] = 1

        return y_h

    def sigmoid(self, V):
        V = V.copy()
        return 1 / (1 + np.exp(-V))

    def sigmoid_dash(self, V):
        V = V.copy()
        s = self.sigmoid(V)
        return s * (1 - s)

    def relU(self, V):
        V = V.copy()
        # V[V < 0] = 0
        return np.maximum(0, V)

    def relU_dash(self, V):
        V = V.copy()

        V[V >= 0] = 1
        V[V < 0] = 0

        # V[V == 0] = self.random_state.uniform(0, 1)
        return V

    def fit(self, x, y):
        try:
            self.start_time = time.time()
            self.train(x, y)
            self.time_to_fit = time.time() - self.start_time
        except KeyboardInterrupt:
            self.time_to_fit = time.time() - self.start_time
            print("Keyboard Interrupt")

    def train(self, x, y):
        eta = self.eta
        m = y.shape[0]
        index = 0
        batch_size = self.mini_batch_size
        max_epoch = self.max_epoch
        epoch = 0
        iteration = 0
        cost_curr = 100
        cost_prev = 50
        sum_cost_prev = 100
        sum_cost_curr = 0

        avg_cost_prev = 100
        avg_cost_curr = 0

        is_converged = False
        count_1000 = 0
        converge_count = 0

        cost_arr = []
        while True:
            if epoch >= max_epoch or is_converged:
                self.n_itr = iteration
                self.n_epoch = epoch
                break
            epoch += 1

            if epoch % 100 == 0:
                print("Epoch:", epoch)

            if self.adaptive:
                eta = self.eta / np.sqrt(epoch)
            x, y = self.shuffle_data(x, y)

            batch_index = 0
            batch_count = m // batch_size

            for batch_index in range(0, batch_count):
                if iteration == 1000 * count_1000:
                    avg_cost_curr = sum_cost_curr / 1000
                    cost_arr.append(avg_cost_curr)
                    self.cost_arr = cost_arr

                    if self.verbose:
                        print(f"Epoch = {epoch}, Avg. cost={avg_cost_curr} Last 1000 iteration avg cost diff= {abs(avg_cost_prev - avg_cost_curr)}")
                    count_1000 += 1
                    sum_cost_curr = 0

                    if self.converge(avg_cost_prev, avg_cost_curr):
                        converge_count += 1
                        print(f"Average cost converged", converge_count)
                        if converge_count >= self.n_epoch_no_change:
                            is_converged = True
                            break
                    else:
                        converge_count = 0

                    avg_cost_prev = avg_cost_curr
                    avg_cost_curr = 0

                iteration += 1
                cost_prev = cost_curr

                x_b = x[batch_index * batch_size: (batch_index + 1) * batch_size]
                y_b = y[batch_index * batch_size: (batch_index + 1) * batch_size]

                a, a_l, z_l = self.feedforward(x_b)

                cost_curr = self.cost(y_b, a)
                sum_cost_curr += cost_curr

                self.backProp(x_b, y_b, a_l, z_l, eta)
        self.cost_arr = cost_arr
        return cost_arr

    def feedforward(self, x):
        '''
        Input:
            x: s*n (s = no. of example, n = no. of features)
            y: s*1 // s*o(Todo)

        Code:
            a: activation, s*u (s = no. of example, u = no. of units in layer)
            a_l : list of activation for each layer,  each activation is s*u (s = no. of example, u = no. of units in layer)
            w: list of weight for each layer, each weight matrix
                is alpha*beta (alpha = no. of unit in layer l, beta = no. of unit in layer l-1)
            b: list of biases, each biases is u*1 (u = no. of units in layer)
        '''
        a = x
        a_l = [x]
        z_l = [x]
        for w, b in zip(self.w[1:-1], self.b[1:-1]):
            z = a@(w.T) + b.T
            # a = self.activation(z)
            a = self.hidden_activation(z)
            a_l.append(a)
            z_l.append(z)

        # last layer
        z = a@(self.w[-1].T) + self.b[-1].T
        a = self.output_activation(z)
        a_l.append(a)
        z_l.append(z)

        return a, a_l, z_l

    def backProp(self, x, y, a_l, z_l, eta):
        '''
        Input:
            x: s*n (s = no. of example, n = no. of features)
            y: s*n_target (s = no. of example, n_target = no. of target class)
            a_l : list of activation for each layer,  each activation is s*u (s = no. of example, u = no. of units in layer)
        '''
        m = y.shape[0]
        # delta_l = dC/dz[l]
        # y_hat = a_l[-1]
        # delta_L = -(y - a_l[-1]) * self.activation_dash(z_l[-1])
        delta_L = (1 / m) * (-(y - a_l[-1])) * self.output_activation_dash(z_l[-1])

        DjDw_L = delta_L.T @ a_l[-2]
        DjDb_L = np.sum(delta_L, axis=0).reshape(-1, 1)

        self.w[-1] -= eta * DjDw_L
        self.b[-1] -= eta * DjDb_L

        delta_l = delta_L

        for index in range(len(self.w) - 2, 0, -1):
            # delta_l = (delta_l @ self.w[index+1]) * self.activation_dash(z_l[index])
            delta_l = (delta_l @ self.w[index + 1]) * self.hidden_activation_dash(z_l[index])

            DjDw_l = delta_l.T @ a_l[index - 1]
            DjDb_l = np.sum(delta_l, axis=0).reshape(-1, 1)

            self.w[index] -= eta * DjDw_l
            self.b[index] -= eta * DjDb_l

    def predict(self, x):
        a, a_l, z_l = self.feedforward(x)
        return np.argmax(a, axis=1).reshape(-1, 1)

    def accuracy(self, actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)

        total = actual.shape[0]

        return (total - np.count_nonzero(actual - predicted)) / total
