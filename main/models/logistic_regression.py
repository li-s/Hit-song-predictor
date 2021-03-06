import numpy as np

from main.utils.evaluation import get_overall_evaluation
from main.visualisation_and_processing.preprocess import normalise


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_weights(dim):
    w = np.random.randn(dim, 1) * 0.1
    b = 0
    return w, b


def propagate(X, Y, w, b):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = - 1 / m * (np.sum((np.dot(Y, np.log(A).T), np.dot((1 - Y), np.log(1 - A).T))))

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(X, Y, w, b)

        dw = grads["dw"]
        db = grads["db"]
        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after {i} iterations: {cost}")

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_weights(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("For training, we have the following metrics:")
    train_accuracy = get_overall_evaluation(Y_prediction_train.T, Y_train.T, X_train, Y_train)
    print()
    print("For test, we have the following metrics:")
    test_accuracy = get_overall_evaluation(Y_prediction_test.T, Y_test.T, X_train, Y_train)

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b}

    return d


if __name__ == "__main__":
    normalized_train_X, train_y, normalized_test_X, test_y = normalise()

    X_train = normalized_train_X.T
    Y_train = train_y.to_numpy()
    Y_train = Y_train.reshape(1, Y_train.shape[0])

    X_test = normalized_test_X.T
    Y_test = test_y.to_numpy()
    Y_test = Y_test.reshape(1, Y_test.shape[0])
    d = model(X_train, Y_train, X_test, Y_test, 1000, 0.1, False)
    w = d["w"]
    b = d["b"]

    print("\nWeights of each feature:")
    for i, val in enumerate(w):
        a = "+" if val[0] > 0 else "-"
        print(f"{i + 1}. Feature: {normalized_train_X.columns[i][0]} | Weight: {a}{round(abs(val[0]), 3)}")
    print(f"\nIntrisic bias {b}")
