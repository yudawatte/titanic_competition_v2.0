"""
Fit Support Vector Classifier model on the preprocessed data
"""
from sklearn.svm import SVC
from helper import read_data, save_data, save_model
from settings import Settings
from model_evaluator import evaluate_model, clf_performance
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

# Read preprocessed data
sett = Settings()
X_train = read_data(sett.PROCESSED_DATA_PATH, sett.PROCESSED_TRAIN_FILENAME)
y_train = read_data(sett.PROCESSED_DATA_PATH, sett.TRAIN_TARGET_SET_FILENAME)
X_test = read_data(sett.PROCESSED_DATA_PATH, sett.PROCESSED_TEST_FILENAME)

print("Read processed data")
print("\tTrain set shape: ",X_train.shape)
print("\tTrain target shape",y_train.shape)
print("\tTest set shape", X_test.shape)

# K neighbors classifier model
svc = SVC(probability=True)

# Evaluate model
print("\nEvaluate model")
evaluate_model(svc, 'Support Vector Classifier', X_train, y_train, 0.3)

# Fine-tuning model with GridSearch
print("\nFine tuning model")
svc = SVC(probability=True)
"""param_grid = [
    {
        'kernel': ['rbf'],
        'gamma': np.arange(0.1, 0.2, 0.01),

        'C': np.arange(0.1, 5.0, 0.1)
    },
    {
        'kernel': ['linear'],
        'C': [.1, 1, 10, 100, 1000]
    },
    {
        'kernel': ['poly'],
        'degree' : [2, 3, 4, 5],
        'C': [.1, 1, 10, 100, 1000]
    }
]"""
param_grid = {'kernel': ['rbf'],
              'gamma': np.arange(0.1, 0.3, 0.01),
              'C': np.arange(0.1, 2.0, 0.01)}
clf_svc = GridSearchCV(svc, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_svc = clf_svc.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_svc, "Support Vector Classifier")

# Predict on test set
predict = best_clf_svc.predict(X_test)

# Saving the model
print("\nSaving Model")
save_model(best_clf_svc, sett.SVC_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.SVC_RESULT_FILENAME,
          index=False, header=True)

"""
Support Vector Classifier
Best Score:	 0.8404494382022472
Best Parameters:	 {'C': 0.6399999999999997, 'gamma': 0.21999999999999995, 'kernel': 'rbf'}

Kaggle score: 0.77511
"""


