from sklearn.ensemble import RandomForestClassifier

from main.utils.evaluation import get_overall_evaluation
from main.visualisation_and_processing.preprocess import normalise


def random_forest(normalized_train_X, train_y, normalized_test_X, test_y):
    random_forest = RandomForestClassifier(n_estimators=40)
    random_forest.fit(normalized_train_X, train_y)
    predict = random_forest.predict(normalized_test_X)

    print("Evaluation of Random Forest Classifier:")
    get_overall_evaluation(predict, test_y, normalized_train_X, train_y, random_forest)


if __name__ == "__main__":
    normalized_train_X, train_y, normalized_test_X, test_y = normalise()
    random_forest(normalized_train_X, train_y, normalized_test_X, test_y)
