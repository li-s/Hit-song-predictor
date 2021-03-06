import sklearn

from main.utils.evaluation import get_overall_evaluation
from main.visualisation_and_processing.preprocess import normalise


def KNN(normalized_train_X, train_y, normalized_test_X, test_y):
    # kNN using bestKVal and bestMetric
    bestKVal = 19
    bestMetric = 'manhattan'
    knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=bestKVal, metric=bestMetric)
    knn_model.fit(normalized_train_X, train_y)
    predict = knn_model.predict(normalized_test_X)
    print(f"Evaluation of kNN Model using k value of {bestKVal} and distance metric of {bestMetric}:")
    get_overall_evaluation(predict, test_y, normalized_train_X, train_y, knn_model)


if __name__ == "__main__":
    normalized_train_X, train_y, normalized_test_X, test_y = normalise()
    KNN(normalized_train_X, train_y, normalized_test_X, test_y)
