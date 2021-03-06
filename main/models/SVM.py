from sklearn.svm import SVC

from main.utils.evaluation import get_overall_evaluation
from main.visualisation_and_processing.preprocess import normalise


def SVM(normalized_train_X, train_y, normalized_test_X, test_y):
    support_vector_classifier = SVC(kernel='poly', degree=2)
    support_vector_classifier.fit(normalized_train_X, train_y)
    predict = support_vector_classifier.predict(normalized_test_X)

    print("Evaluation of Support Vector Classifier:")
    get_overall_evaluation(predict, test_y, normalized_train_X, train_y)


if __name__ == "__main__":
    normalized_train_X, train_y, normalized_test_X, test_y = normalise()
    SVM(normalized_train_X, train_y, normalized_test_X, test_y)
