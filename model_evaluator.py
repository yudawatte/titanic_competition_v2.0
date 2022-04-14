"""
Contain functionalities which will be used for evaluating trained models.
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, model_name, dataset, target, test_size=0.3, cv=10, heat_map=False):
    # Split train sets, test sets
    X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=test_size, shuffle=True, random_state=None)

    print("Evaluate - ", model_name)
    model.fit(X_train, y_train.values.ravel())

    predictions = model.predict(X_test)
    print("\tAccuracy: ", round(accuracy_score(predictions, y_test.values.ravel()) * 100, 2))
    result_cv = cross_val_score(model, dataset, target.values.ravel(), cv=cv, scoring='accuracy')
    if heat_map:
        print('\tMean Cross Validated Score: ', round(result_cv.mean() * 100, 2))
        y_pred = cross_val_predict(model, dataset, target.values.ravel(), cv=cv)
        sns.heatmap(confusion_matrix(target, y_pred), annot=True, fmt='3.0f', cmap="summer")
        plt.title('Confusion_matrix', y=1.05, size=15)

def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score:\t', str(classifier.best_score_))
    print('Best Parameters:\t ' + str(classifier.best_params_))

