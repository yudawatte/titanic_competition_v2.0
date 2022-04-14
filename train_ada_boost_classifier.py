"""
Fit Ada Boost Classifier model on the preprocessed data
"""
from sklearn.ensemble import AdaBoostClassifier
from helper import read_data, save_data, save_model
from settings import Settings
from model_evaluator import evaluate_model, clf_performance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
ada = AdaBoostClassifier()

# Evaluate model
print("\nEvaluate model")
evaluate_model(ada, 'Ada Boost Classifier', X_train, y_train, 0.3)

# Fine-tuning model with GridSearch
print("\nFine tuning model")
ada = AdaBoostClassifier()
"""
# First do a high level tune-up with RandomizedSearchCV
param_grid = {
    'n_estimators': [550, 525, 600, 625, 650, 675],
    'learning_rate':[0.001, 0.01, 0.1, 1],
    'algorithm': ['SAMME', 'SAMME.R'],
}
clf_ada = RandomizedSearchCV(ada, param_distributions=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_ada = clf_ada.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_ada, "Ada Boost Classifier")"""

"""
RandomizedSearchCV
Best Score:	 0.8235955056179775
Best Parameters:	 {'n_estimators': 600, 'learning_rate': 0.1, 'algorithm': 'SAMME'}
"""
param_grid = {'n_estimators': np.arange(590, 610, 1),
              'learning_rate': np.arange(0.01, 0.2, 0.01),
               'algorithm': ['SAMME']}
clf_ada = GridSearchCV(ada, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_ada = clf_ada.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_ada, "Ada Boost Classifier")

# Predict on test set
predict = best_clf_ada.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(best_clf_ada, sett.ADA_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.ADA_RESULT_FILENAME,
          index=False, header=True)

"""
Ada Boost Classifier
Best Score:	 0.8258426966292134
Best Parameters:	 {'algorithm': 'SAMME', 'learning_rate': 0.18000000000000002, 'n_estimators': 590}
"""


