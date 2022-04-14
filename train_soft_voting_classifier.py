"""
Fit trained models with a voting classifier (soft voting)
"""
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from helper import read_data, save_data, save_model, load_model, show_results
from settings import Settings
from model_evaluator import evaluate_model, clf_performance
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
ada = load_model(sett.ADA_MODEL_NAME)
gbc = load_model(sett.GBC_MODEL_NAME)

# Since ridge classifier has no 'predict_proba' attribute
rc_calb = CalibratedClassifierCV(rc)

# Voting classifier model
print("\nSoft voting classifier")
"""voting_clf = VotingClassifier(
    estimators = [('neive bayes',gnb),
                  ('log reg',lr),
                  ('ridge calb',rc_calb),
                  ('linear dis ana',lda),
                  ('knn',knn),
                  ('svc',svc),
                  ('random forest',rf),
                  ('ada boost',ada),
                  ('gradient boost',gbc)],
    voting = 'soft')"""

"""
print("\nSoft voting classifier with best 4 models")
voting_clf_b4 = VotingClassifier(
    estimators = [('knn',knn),
                  ('svc',svc),
                  ('random forest',rf),
                  ('gradient boost',gbc)],
    voting = 'soft', n_jobs=-1, verbose=True)

# Gradient boost model taking comparatively longer time to train on 
# Soft voting classifier    
"""
print("soft voting with all models except gradient boost classifier")
voting_clf_b8 = VotingClassifier(
    estimators = [('neive bayes',gnb),
                  ('log reg',lr),
                  ('ridge calb',rc_calb),
                  ('linear dis ana',lda),
                  ('knn',knn),
                  ('svc',svc),
                  ('random forest',rf),
                  ('ada boost',ada)],
    voting = 'soft')


# Evaluate model
print("\nEvaluate model")
#cv = cross_val_score(voting_clf_b4, X_train, y_train.values.ravel(), cv=5)
cv = cross_val_score(voting_clf_b8, X_train, y_train.values.ravel(), cv=5)
show_results(cv)

#voting_clf_b4.fit(X_train, y_train.values.ravel())
voting_clf_b8.fit(X_train, y_train.values.ravel())
#clf_performance(best_clf_vc, "Vector Classifier")

# Predict on test set
#predict = voting_clf_b4.predict(X_test)
predict = voting_clf_b8.predict(X_test)

# Saving the model
print("\nSaving Model")
#save_model(voting_clf, sett.VC_MODEL_NAME)
#save_model(voting_clf_b4, "vc_model_b4.sav")
save_model(voting_clf_b8, "vc_model_b4.sav")

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
#save_data(base_submission, sett.RESULT_DATA_PATH, sett.VC_RESULT_FILENAME,
#save_data(base_submission, sett.RESULT_DATA_PATH, "vc_results_b4.csv",
save_data(base_submission, sett.RESULT_DATA_PATH, "vc_results_b8.csv",
          index=False, header=True)

print("Prediction duration: %s seconds."%(time.time()-start_time))

"""
vc_results_b4
scross validation scores:  [0.83146067 0.79213483 0.83146067 0.80337079 0.87078652]
scross validation mean:  0.8258426966292134
Kaggle Score: 0.75598
"""

"""
vc_results_b8
scross validation scores:  [0.82022472 0.80898876 0.8258427  0.80337079 0.85955056]
scross validation mean:  0.8235955056179776
Kaggle Score: 0.77272
"""