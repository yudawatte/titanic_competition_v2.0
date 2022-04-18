"""
Fit Gradient Boosting Classifier model on the preprocessed data
"""
from sklearn.ensemble import GradientBoostingClassifier
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

# Gradient Boosting classifier model
gbc = GradientBoostingClassifier()

# Evaluate model
print("\nEvaluate model")
evaluate_model(gbc, 'Gradient Boosting Classifier', X_train, y_train, 0.3)

# Fine-tuning model with GridSearch
print("\nFine tuning model")
gbc = GradientBoostingClassifier()

# First do a high level tune-up with RandomizedSearchCV
"""param_grid = {
    'loss': ['deviance', 'exponential'],
    'learning_rate':[0.001, 0.01, 0.1, 1],
    'n_estimators': [450, 500, 550, 600,650],
    'criterion':['friedman_mse', 'squared_error'],
    #'criterion':['friedman_mse'],
    'min_samples_split':[2, 3],
    #'min_samples_split':[2],
    'min_samples_leaf':[2, 3],
    #'min_samples_leaf':[3],
    #'max_depth': [15, 20, 25],
    'max_depth': [20, 25, 30, 35],
    'max_features':['auto', 'sqrt', 'log2', None]
    #'max_features':['auto']
}
clf_gbc = RandomizedSearchCV(gbc, param_distributions=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_gbc = clf_gbc.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_gbc, "Gradient Boosting Classifier")"""

"""
Gradient Boosting Classifier
Best Score:	 0.8224719101123595
Best Parameters:	 {'n_estimators': 650, 
                      'min_samples_split': 3, 
                      'min_samples_leaf': 3, 
                      'max_features': 'log2', 
                      'max_depth': 20, 
                      'loss': 'exponential', 
                      'learning_rate': 0.001, 
                      'criterion': 'squared_error'}
"""
param_grid = {'loss': ['exponential'],
              'learning_rate': np.arange(0.0005, 0.0015, 0.0001),
              'n_estimators': np.arange(640,660, 1),
              'criterion':['squared_error'],
              'min_samples_split':[2,3],
              'min_samples_leaf':[2,3],
              'max_depth': np.arange(15, 25, 1),
              'max_features':['sqrt', 'log2']}
clf_gbc = GridSearchCV(gbc, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_gbc = clf_gbc.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_gbc, "Gradient Boosting Classifier")

# Predict on test set
predict = best_clf_gbc.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(best_clf_gbc, sett.GBC_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.GBC_RESULT_FILENAME,
          index=False, header=True)

"""
Gradient Boosting Classifier
Best Score:	 0.8314606741573034
Best Parameters:	 {'criterion': 'friedman_mse', 
                      'learning_rate': 0.01, 
                      'loss': 'exponential', 
                      'max_depth': 25, 
                      'max_features': 'sqrt', 
                      'min_samples_leaf': 3, 
                      'min_samples_split': 2, 
                      'n_estimators': 548}
"""


