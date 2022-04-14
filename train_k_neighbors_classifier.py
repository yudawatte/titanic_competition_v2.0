"""
Fit K Neighbors Classifier model on the preprocessed data
"""
from sklearn.neighbors import KNeighborsClassifier
from helper import read_data, save_data, save_model
from settings import Settings
from model_evaluator import evaluate_model, clf_performance
from sklearn.model_selection import GridSearchCV
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
knn = KNeighborsClassifier()

# Evaluate model
print("\nEvaluate model")
evaluate_model(knn, 'K Neighbors Classifier', X_train, y_train, 0.3)

# Fine-tuning model with GridSearch
print("\nFine tuning model")
knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : [8,9,10,11,12,13,14],
              #'n_neighbors' : [3,5,7,9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree', 'brute'],
              'p' : [1,2]}
clf_knn = GridSearchCV(knn, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_knn = clf_knn.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_knn, "KNeigbors Classifier")

# Predict on test set
predict = best_clf_knn.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(best_clf_knn, sett.KNN_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.KNN_RESULT_FILENAME,
          index=False, header=True)

"""
KNeigbors Classifier
Best Score:	 0.8382022471910112
Best Parameters:	 {'algorithm': 'kd_tree', 'n_neighbors': 11, 'p': 1, 'weights': 'uniform'}
"""


