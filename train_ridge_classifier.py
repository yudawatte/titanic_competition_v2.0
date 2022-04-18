"""
Fit Ridge Classifier model on the preprocessed data
"""
from sklearn.linear_model import RidgeClassifier
from helper import read_data, save_data, save_model
from settings import Settings
from model_evaluator import evaluate_model, clf_performance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np

# Read preprocessed data
sett = Settings()
X_train = read_data(sett.PROCESSED_DATA_PATH, sett.PROCESSED_TRAIN_FILENAME)
y_train = read_data(sett.PROCESSED_DATA_PATH, sett.TRAIN_TARGET_SET_FILENAME)
X_test = read_data(sett.PROCESSED_DATA_PATH, sett.PROCESSED_TEST_FILENAME)

print("Read processed data")
print("\tTrain set shape: ",X_train.shape)
print("\tTrain target shape",y_train.shape)
print("\tTest set shape", X_test.shape)

# Ridge classifier model
rc = RidgeClassifier()

# Evaluate model
print("\nEvaluate model")
evaluate_model(rc, 'Ridge Classifier', X_train, y_train, 0.3)

# Fine-tuning model with GridSearch
print("\nFine tuning model")
rc = RidgeClassifier()

"""param_grid = [
    {
    	'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
        'alpha': [0.01, 0.1, 1, 10, 100],
        'max_iter': [10000],
    	'tol': [1e-4]
	},
	{
    	'solver': ['sag', 'saga'],
        'alpha': [0.01, 0.1, 1, 10, 100],
        'max_iter': [10000],
    	'tol': [1e-4],
    	'random_state': [42]
	},
	{
    	'solver': ['lbfgs'],
        'alpha': [0.01, 0.1, 1, 10, 100],
        'max_iter': [10000],
    	'tol': [1e-4],
        'positive':[True],
    	'random_state': [42]
	}
]

clf_rc = RandomizedSearchCV(rc, param_distributions=param_grid,
                            cv=5, verbose=True, n_jobs=-1)

best_clf_rc = clf_rc.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_rc, "Ridge Classifier")"""

"""
Ridge Classifier - Randomized Search CV best results
Best Score:	 0.8280898876404494
Best Parameters:	 {'tol': 0.0001, 'solver': 'sparse_cg', 'max_iter': 10000, 'alpha': 0.01}
"""

param_grid = {
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
    'alpha': np.arange(0 ,5, 0.01),
    'max_iter': [10000],
    'tol': [1e-4]
}
clf_rc = GridSearchCV(rc, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_rc = clf_rc.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_rc, "Ridge Classifier")

# Predict on test set
predict = best_clf_rc.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(best_clf_rc, sett.RC_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.RC_RESULT_FILENAME, index=False, header=True)

"""
Ridge Classifier - GridSearch best results
Best Score:	 0.8292134831460676 
Best Parameters:	 {'alpha': 2.49, 'max_iter': 10000, 'solver': 'sparse_cg', 'tol': 0.0001}
"""
