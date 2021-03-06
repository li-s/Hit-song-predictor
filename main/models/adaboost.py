from sklearn.ensemble import AdaBoostClassifier

from main.utils.evaluation import get_overall_evaluation
from main.visualisation_and_processing.preprocess import normalise


def adaboost(normalized_train_X, train_y, normalized_test_X, test_y):
    ada_classifier = AdaBoostClassifier(n_estimators=40, random_state=0)
    ada_classifier.fit(normalized_train_X, train_y)
    predict = ada_classifier.predict(normalized_test_X)

    print("Evaluation of Adaboost Classifier:")
    get_overall_evaluation(predict, test_y, normalized_train_X, train_y, ada_classifier)


if __name__ == "__main__":
    normalized_train_X, train_y, normalized_test_X, test_y = normalise()
    adaboost(normalized_train_X, train_y, normalized_test_X, test_y)
