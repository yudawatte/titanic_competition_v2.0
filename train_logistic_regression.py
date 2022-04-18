"""
Fit Logistic Regression model on the preprocessed data
"""
from sklearn.linear_model import LogisticRegression
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

# Logistic regression model
lr = LogisticRegression()

# Evaluate model
print("\nEvaluate model")
evaluate_model(lr, 'Logistic Regression', X_train, y_train, 0.3)

# Fine-tuning model with RandomizedSearchCV
print("\nFine tuning model")
lr = LogisticRegression()

"""param_grid = [
    {
    	'solver': ['newton-cg', 'lbfgs', 'sag'],
    	'penalty':['l2', 'none'],
    	'tol': [1e-4],
        'max_iter': [10000],
    	'C': [0.01, 0.1, 1, 10, 100],
    	'random_state': [42],
    	'multi_class': ['auto', 'ovr'],
    	'n_jobs': [-1]
	},
	{
    	'solver': ['liblinear'],
    	'penalty':['l1', 'l2'],
    	'tol': [1e-4],
        'max_iter': [10000],
    	'C': [0.01, 0.1, 1, 10, 100],
    	'random_state': [42],
    	'multi_class': ['auto', 'ovr'],
    	'n_jobs': [-1]
	},
	{
    	'solver': ['saga'],
    	'penalty':['l1', 'l2', 'none'],
    	'tol': [1e-4],
        'max_iter': [10000],
    	'C': [0.01, 0.1, 1, 10, 100],
    	'random_state': [42],
    	'multi_class': ['auto', 'ovr'],
    	'n_jobs': [-1]
	},
	{
    	'solver': ['saga'],
    	'penalty':['elasticnet'],
    	'tol': [1e-4],
        'max_iter': [10000],
    	'C': [0.01, 0.1, 1, 10, 100],
    	'random_state': [42],
    	'multi_class': ['auto', 'ovr'],
    	'n_jobs': [-1],
	    'l1_ratio': [0, 0.2, 0.5, 0.7, 1]
	}
]

clf_lr = RandomizedSearchCV(lr, param_distributions=param_grid,
                            cv=5, verbose=True, n_jobs=-1)

best_clf_lr = clf_lr.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_lr, "Logistic Regression")"""

"""
Logistic Regression - Randomized Search CV best results
Best Score:	 0.8269662921348315
Best Parameters:	 {'tol': 0.0001, 
                      'solver': 'newton-cg', 
                      'random_state': 42, 
                      'penalty': 'l2', 
                      'n_jobs': -1, 
                      'multi_class': 'auto', 
                      'max_iter': 10000, 
                      'C': 1}
"""

param_grid = {'solver': ['newton-cg', 'lbfgs', 'sag'],
              'tol': [1e-4],
              'penalty':['l2'],
              'max_iter': [10000],
              'C': np.arange(0.1, 2.5, 0.1),
              'random_state': [42],
              'multi_class': ['auto', 'ovr'],
              'n_jobs': [-1]
              }
clf_lr = GridSearchCV(lr, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_lr = clf_lr.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_lr, "Logistic Regression")

# Predict on test set
predict = best_clf_lr.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(best_clf_lr, sett.LR_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.LR_RESULT_FILENAME,
          index=False, header=True)

"""
Logistic Regression - GridSearch best results
Best Score:	 0.8292134831460674
Best Parameters:	 {'C': 0.4, 
                      'max_iter': 10000, 
                      'multi_class': 'auto', 
                      'n_jobs': -1, 
                      'penalty': 'l2', 
                      'random_state': 42, 
                      'solver': 'newton-cg', 
                      'tol': 0.0001}
"""