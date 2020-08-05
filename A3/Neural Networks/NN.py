#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import time
import joblib
from joblib import Parallel, delayed
import os
import sys
from NNClassifier import NNClassifier
from sklearn.metrics import f1_score


# Time taken by function
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__ + " took " + str(end - start) + " s")
        return result
    return wrapper


train_path = "Alphabets/train.csv"
test_path = "Alphabets/test.csv"


def f1(y_true, y_pred):
    return f1_score(y_true.reshape(-1,), y_pred.reshape(-1,), average="micro")


def encodeY(y, n_target=26):
    y_h = np.zeros((y.shape[0], n_target))
    for i, item in enumerate(y):
        y_h[i][item[0]] = 1
    return y_h


def normalize(V):
    return V / 255


def readData(path):
    data = np.array(pd.read_csv(path, header=None))
    x = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    return (x, y)


def displayImg(x):
    plt.imshow(x.reshape(-1, 28))


train_x, train_y = readData(train_path)
test_x, test_y = readData(test_path)


def fileExist(file_name):
    return os.path.isfile(file_name)


def plotCosts(model_arr):
    for index, model in enumerate(model_arr[:]):
        plt.title(f"Cost vs No. of epochs, hidden_layer={model.hidden_layers}", weight="bold")
        plt.xlabel("Epoch", size=12)
        plt.ylabel("Cost", size=12)

        y = model.cost_arr[1:]
        x = np.arange(0, len(y) * 1000, 1000) / 130
        plt.plot(x, y, label='Train cost')

        plt.legend()
        plt.show()


def partBC(model_name=None, eta=0.1, adaptive=False):

    hidden_layer_unit = [100, 50, 10, 5, 1]
    model_arr = []
    if not fileExist(model_name):
        for u in hidden_layer_unit:
            print(u)
            model = NNClassifier(mini_batch_size=100, hidden_layers=[u], n_target=26, eta=eta, hidden_activation="sigmoid", adaptive=adaptive, random_state=2)
            model.fit(normalize(train_x), encodeY(train_y))
            model_arr.append(model)
        joblib.dump(model_arr, model_name)

    model_arr = joblib.load(model_name)

    train_accuracies = []
    test_accuracies = []
    time_arr = []

    for model, unit in zip(model_arr, hidden_layer_unit):
        train_score = model.accuracy(train_y, model.predict(normalize(train_x)))
        test_score = model.accuracy(test_y, model.predict(normalize(test_x)))
        train_accuracies.append(train_score)
        test_accuracies.append(test_score)
        time_arr.append(model.time_to_fit)
        train_f1 = f1(train_y, model.predict(normalize(train_x)))
        test_f1 = f1(test_y, model.predict(normalize(test_x)))

        print(f"[{unit}] Train Accuracy: {round(train_score*100, 4)}, Test Accuracy: {round(test_score*100, 4)}, Train Time: {round(model.time_to_fit, 3)}, Epoch: {model.n_epoch}, Iteration: {model.n_itr}, Final cost:{model.cost_arr[-1]} Train F1: {round(train_f1, 4)}, Test F1: {round(test_f1, 4)}")

    # Accuracy plot
    plt.title("No. of hidden units vs Accuracy", weight="bold")
    plt.xlabel("No. of hidden unit", size=12)
    plt.ylabel("Accuracy(%)", size=12)
    plt.plot(hidden_layer_unit, np.array(train_accuracies) * 100, linestyle='--', marker='o', color='b', label="Train accuracy")
    plt.plot(hidden_layer_unit, np.array(test_accuracies) * 100, linestyle='--', marker='o', color='g', label="Test accuracy")
    plt.legend()
    plt.show()

    # Time plot
    plt.title("No. of hidden units vs Time to fit", weight="bold")
    plt.xlabel("No. of hidden unit", size=12)
    plt.ylabel("Time(sec)", size=12)
    plt.plot(hidden_layer_unit, np.array(time_arr), linestyle='--', marker='o', color='b', label="Train time")
    plt.legend()
    plt.show()

    # Cost plot
    plotCosts(model_arr)


@time_it
def partB():
    partBC(model_name="nnModel_arr_b", eta=0.1, adaptive=False)


@time_it
def partC():
    partBC(model_name="nnModel_arr_c", eta=0.5, adaptive=True)


@time_it
def partD():

    hidden_layers = [100, 100]

    model_name_relu = "nnModel_d_relu"
    if not fileExist(model_name_relu):
        model_relu = NNClassifier(mini_batch_size=100, hidden_layers=hidden_layers, n_target=26, eta=0.3, hidden_activation="relu", adaptive=True, random_state=2, verbose=True)
        model_relu.fit(normalize(train_x), encodeY(train_y))
        joblib.dump(model_relu, model_name_relu)
    model_relu = joblib.load(model_name_relu)

    model_name_sig = "nnModel_d_sig"
    if not fileExist(model_name_sig):
        model_sig = NNClassifier(mini_batch_size=100, hidden_layers=hidden_layers, n_target=26, eta=0.3, hidden_activation="sig", adaptive=True, random_state=2, verbose=True)
        model_sig.fit(normalize(train_x), encodeY(train_y))
        joblib.dump(model_sig, model_name_sig)
    model_sig = joblib.load(model_name_sig)

    # Accuracy comparison
    train_acc_relu = model_relu.accuracy(train_y, model_relu.predict(normalize(train_x)))
    test_acc_relu = model_relu.accuracy(test_y, model_relu.predict(normalize(test_x)))

    train_acc_sig = model_sig.accuracy(train_y, model_sig.predict(normalize(train_x)))
    test_acc_sig = model_sig.accuracy(test_y, model_sig.predict(normalize(test_x)))

    # print("train accuracy, test accuracy")
    # print("relU", train_acc_relu, test_acc_relu)
    # print("Sigmoid", train_acc_sig, test_acc_sig)
    print(f"[{hidden_layers}, RelU] Train Accuracy: {round(train_acc_relu * 100, 4)}, Test Accuracy: {round(test_acc_relu * 100, 4)}, Train Time: {round(model_relu.time_to_fit , 3)}, Epoch: {model_relu.n_epoch}, Iteration: {model_relu.n_itr} Final Cost: {model_relu.cost_arr[-1]}")
    print(f"[{hidden_layers}, Sigmoid] Train Accuracy: {round(train_acc_sig * 100, 4)}, Test Accuracy: {round(test_acc_sig * 100, 4)}, Train Time: {round(model_sig.time_to_fit, 3)}, Epoch: {model_sig.n_epoch}, Iteration: {model_sig.n_itr} Final Cost: {model_sig.cost_arr[-1]}")

    # cost comparison
    plt.title("Cost: RelU vs  Sigmoid", weight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")

    y = model_relu.cost_arr[1:]
    x = np.arange(0, len(y) * 1000, 1000) / 130

    plt.plot(x, model_relu.cost_arr[1:], label="RelU")
    plt.plot(x, model_sig.cost_arr[1:], label="Sigmoid")
    plt.legend()

    plt.show()


@time_it
def partE():
    model_name_relu = "mlp_relu1"
    model_name_sig = "mlp_sig"

    if not fileExist(model_name_relu):
        mlp_relu = MLPClassifier(
            hidden_layer_sizes=(100, 100),
            activation='relu',
            solver='sgd',
            batch_size=100,
            learning_rate='invscaling',
            learning_rate_init=0.5,
            power_t=0.5,
            max_iter=1500,
            shuffle=True,
            random_state=1,
            tol=1e-7,
            momentum=0,
            # verbose=True
        )
        s = time.time()
        mlp_relu.fit(normalize(train_x), encodeY(train_y))
        e = time.time() - s
        print(f"Time taken for MLP relu = {e}")
        mlp_relu.time_to_fit = e
        # joblib.dump(mlp_relu, model_name_relu)

    if not fileExist(model_name_sig):
        mlp_sig = MLPClassifier(
            hidden_layer_sizes=(100, 100),
            activation='logistic',
            solver='sgd',
            batch_size=100,
            learning_rate='invscaling',
            learning_rate_init=0.3,
            power_t=0.5,
            max_iter=1500,
            shuffle=True,
            random_state=1,
            tol=1e-7,
            # verbose=1
        )
        s = time.time()
        mlp_sig.fit(normalize(train_x), train_y.reshape(-1,))
        e = time.time() - s
        print(f"Time taken for MLP sigmoid = {e}")
        mlp_sig.time_to_fit = e
        # joblib.dump(mlp_sig, model_name_sig)

    mlp_relu = joblib.load(model_name_relu)
    mlp_sig = joblib.load(model_name_sig)

    relu_score_train = getMLPAccuracy(train_y, mlp_relu.predict_proba(normalize(train_x)))
    relu_score_test = getMLPAccuracy(test_y, mlp_relu.predict_proba(normalize(test_x)))

    sig_score_train = getMLPAccuracy(train_y, mlp_sig.predict_proba(normalize(train_x)))
    sig_score_test = getMLPAccuracy(test_y, mlp_sig.predict_proba(normalize(test_x)))

    print(f"(100,100), RelU] Train Accuracy: {round(relu_score_train * 100, 4)}, Test Accuracy: {round(relu_score_test * 100, 4)}, Train Time: {round(mlp_relu.time_to_fit * 0.75, 3)}, Iteration: {mlp_relu.n_iter_}, Final cost: {mlp_relu.loss_curve_[-1]}")
    print(f"(100,100), Sigmoid] Train Accuracy: {round(sig_score_train * 100, 4)}, Test Accuracy: {round(sig_score_test * 100, 4)}, Train Time: {round(mlp_sig.time_to_fit * 0.75, 3)}, Iteration: {mlp_sig.n_iter_}, Final cost: {mlp_sig.loss_curve_[-1]}")

    plt.title("Cost: RelU", weight="bold")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.plot(mlp_relu.loss_curve_, label="Relu")
    plt.legend()
    plt.show()

    plt.title("Cost: Sigmoid", weight="bold")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.plot(mlp_sig.loss_curve_, label="Sigmoid", c="r")
    plt.legend()
    plt.show()


def getMLPAccuracy(actual, prob):
    a = np.argmax(prob, axis=1).reshape(-1, 1)
    predicted = np.array(a)
    total = actual.shape[0]
    return (total - np.count_nonzero(actual - predicted)) / total


if __name__ == "__main__":
    try:
        if len(sys.argv) > 2:
            raise "Invalid parameter "
        if len(sys.argv) == 1 or sys.argv[1] == "b":
            print("-" * 70)
            print("Part B running...")
            partB()
        if len(sys.argv) == 1 or sys.argv[1] == "c":
            print("-" * 70)
            print("Part C running...")
            partC()
        if len(sys.argv) == 1 or sys.argv[1] == "d":
            print("-" * 70)
            print("Part D running...")
            partD()
        if len(sys.argv) == 1 or sys.argv[1] == "e":
            print("-" * 70)
            print("Part E running...")
            partE()
    except:
        print("Invalid parameter\n 'python3 NN.py' or for running part b only 'python3 NN.py b' ")
