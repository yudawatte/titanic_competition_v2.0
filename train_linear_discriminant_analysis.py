"""
Fit Linear Discriminant Analysis model on the preprocessed data
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# Linear discriminant analysis model
lda = LinearDiscriminantAnalysis()

# Evaluate model
print("\nEvaluate model")
evaluate_model(lda, 'Linear Discriminant Analysis', X_train, y_train, 0.3)

# Fine-tuning model with GridSearch
print("\nFine tuning model")
lda = LinearDiscriminantAnalysis()

"""param_grid = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [None, 'auto', 0.1, 0.3, 0.5, 0.7],
}
clf_lda = RandomizedSearchCV(lda, param_distributions=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_lda = clf_lda.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_lda, "Linear Discriminant Analysis")"""

param_grid = {'solver': ['svd', 'lsqr']}#,
              #'shrinkage': ['auto']}
clf_lda = GridSearchCV(lda, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_lda = clf_lda.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_lda, "Linear Discriminant Analysis")

# Predict on test set
predict = best_clf_lda.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(best_clf_lda, sett.LDA_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.LDA_RESULT_FILENAME,
          index=False, header=True)

"""
Linear Discriminant Analysis
Best Score:	 0.8269662921348315
Best Parameters:	 {'shrinkage': None, 'solver': 'svd'}
"""