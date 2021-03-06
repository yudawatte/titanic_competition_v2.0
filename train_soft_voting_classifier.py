"""
Fit trained models with a voting classifier (soft voting)
"""
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from helper import read_data, save_data, save_model, load_model, show_results
from settings import Settings
from model_evaluator import evaluate_model, clf_performance
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

start_time = time.time()
# Read preprocessed data
sett = Settings()
X_train = read_data(sett.PROCESSED_DATA_PATH, sett.PROCESSED_TRAIN_FILENAME)
y_train = read_data(sett.PROCESSED_DATA_PATH, sett.TRAIN_TARGET_SET_FILENAME)
X_test = read_data(sett.PROCESSED_DATA_PATH, sett.PROCESSED_TEST_FILENAME)

print("Read processed data")
print("\tTrain set shape: ",X_train.shape)
print("\tTrain target shape",y_train.shape)
print("\tTest set shape", X_test.shape)

# Load saved best models
print("\nLoading saved models")
sett = Settings()
gnb = load_model(sett.GNB_MODEL_NAME)
lr = load_model(sett.LR_MODEL_NAME)
rc = load_model(sett.RC_MODEL_NAME)
lda = load_model(sett.LDA_MODEL_NAME)
knn = load_model(sett.KNN_MODEL_NAME)
svc = load_model(sett.SVC_MODEL_NAME)
rf = load_model(sett.RF_MODEL_NAME)
#ada = load_model(sett.ADA_MODEL_NAME)
#gbc = load_model(sett.GBC_MODEL_NAME)

# Since ridge classifier has no 'predict_proba' attribute
rc_calb = CalibratedClassifierCV(rc)

# Voting classifier model
print("\nSoft voting classifier")

voting_clf = VotingClassifier(
    estimators = [('neive bayes',gnb),
                  ('log reg',lr),
                  ('ridge calb',rc_calb),
                  ('linear dis ana',lda),
                  ('knn',knn),
                  ('svc',svc),
                  ('random forest',rf)],
    verbose=True,n_jobs=-1)

param_grid = {
    'voting': ['hard', 'soft'],
}

clf_rc = GridSearchCV(rc, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_rc = clf_rc.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_rc, "Voting Classifier")

# Predict on test set
predict = best_clf_rc.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(best_clf_rc, sett.VC_MODEL_NAME)

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, sett.VC_RESULT_FILENAME, index=False, header=True)