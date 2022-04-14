"""
Fit Gaussian Naive Bayes model on the preprocessed data
"""
from sklearn.naive_bayes import GaussianNB
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
gnb = GaussianNB()

# Evaluate model
print("\nEvaluate model")
evaluate_model(gnb, 'Gaussian Naive Bayes Classifier', X_train, y_train, 0.3)

# Train the model on test set
gnb.fit(X_train, y_train.values.ravel())

# Predict on test set
predict = gnb.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(gnb, sett.GNB_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.GNB_RESULT_FILENAME,
          index=False, header=True)

"""
Gaussian Naive Bayes Classifier
Accuracy:  76.03
Mean Cross Validated Score:  76.63
"""


