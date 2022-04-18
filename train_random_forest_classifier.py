"""
Fit Random Forest Classifier model on the preprocessed data
"""
from sklearn.ensemble import RandomForestClassifier
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
rf = RandomForestClassifier()

# Evaluate model
print("\nEvaluate model")
evaluate_model(rf, 'Random Forest Classifier', X_train, y_train, 0.3)

# Fine-tuning model with GridSearch
print("\nFine tuning model")
rf = RandomForestClassifier()
"""param_grid = {
    'n_estimators': [200,300,400,500, 600],
    'criterion':['gini','entropy'],
    'bootstrap': [True, False],
    'max_depth': [15, 20, 25],
    'max_features': ['auto','sqrt', 10, 20, 30],
    'min_samples_leaf': [2,3],
    'min_samples_split': [2,3],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}
clf_rf = RandomizedSearchCV(rf, param_distributions=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_rf = clf_rf.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_rf, "Random Forest Classifier")"""
"""
Random Forest Classifier - RandomizedSearchCV results
Best Score:	 0.8303370786516855
Best Parameters:	 {'n_estimators': 300, 
                      'min_samples_split': 3, 
                      'min_samples_leaf': 3, 
                      'max_features': 20, 
                      'max_depth': 15, 
                      'criterion': 'entropy', 
                      'class_weight': 'balanced', 
                      'bootstrap': True}
"""
param_grid = {'n_estimators': np.arange(290, 310, 1),
              'criterion':['entropy'],
              'bootstrap': [True],
              'max_depth': np.arange(10, 20, 1),
              'max_features': np.arange(15, 25, 1),
              'min_samples_leaf': [2, 3],
              'min_samples_split': [2, 3]}
clf_rf = GridSearchCV(rf, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_rf = clf_rf.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_rf, "Random Forest Classifier")

# Predict on test set
predict = best_clf_rf.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(best_clf_rf, sett.RF_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.RF_RESULT_FILENAME,
          index=False, header=True)

"""
Random Forest Classifier - GridSearch best results
Best Score:	 0.8359550561797754
Best Parameters:	 {'bootstrap': True, 
                      'criterion': 'entropy', 
                      'max_depth': 12, 
                      'max_features': 24, 
                      'min_samples_leaf': 2, 
                      'min_samples_split': 2, 
                      'n_estimators': 303}
"""


