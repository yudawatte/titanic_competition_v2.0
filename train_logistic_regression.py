"""
Fit Logistic Regression model on the preprocessed data
"""
from sklearn.linear_model import LogisticRegression
from helper import read_data, save_data, save_model
from settings import Settings
from model_evaluator import evaluate_model
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

# Train the model on test set
lr.fit(X_train, y_train.values.ravel())

# Predict on test set
predict = lr.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(lr, sett.LR_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.LR_RESULT_FILENAME, index=False, header=True)

"""
Logistic Regression
Accuracy:  81.27
Mean Cross Validated Score:  82.7
"""