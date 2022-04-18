"""
Fit Ada Boost Classifier model on the preprocessed data
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from helper import read_data, save_data, save_model, load_model
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
ada = AdaBoostClassifier()

# Evaluate model
print("\nEvaluate model")
evaluate_model(ada, 'Ada Boost Classifier', X_train, y_train, 0.3)


# AdaBoost with VotingClassifier - hard
#vc_hard = load_model("vc_hard_model.sav")
"""knn = KNeighborsClassifier(algorithm='ball_tree',
                           leaf_size=23,
                           n_jobs=-1,
                           n_neighbors=11,
                           p=2,
                           weights='uniform')"""
"""svc = SVC(C=1.7000000000000002,
          gamma='scale',
          kernel='rbf',
          max_iter=10000,
          tol=1e-4)"""
rf = RandomForestClassifier(bootstrap=True,
                            criterion='entropy',
                            max_depth=12,
                            max_features=24,
                            min_samples_leaf=2,
                            min_samples_split=2,
                            n_estimators=303)

ada = AdaBoostClassifier(base_estimator=rf)

# Fine-tuning model
print("\nFine tuning model")

# First do a high level tune-up with RandomizedSearchCV
"""param_grid = {
    'n_estimators': [100, 300, 500, 600, 700],
    'learning_rate':[0.001, 0.01, 0.1, 1],
    'algorithm': ['SAMME', 'SAMME.R'],
}
clf_ada = RandomizedSearchCV(ada, param_distributions=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_ada = clf_ada.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_ada, "Ada Boost Classifier")"""

param_grid = {'n_estimators': np.arange(290, 310, 1),
              'learning_rate': np.arange(0.01, 0.2, 0.01),
              'algorithm': ['SAMME']}
clf_ada = GridSearchCV(ada, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_ada = clf_ada.fit(X_train, y_train.values.ravel())
clf_performance(best_clf_ada, "Ada Boost Classifier")

# Predict on test set
predict = best_clf_ada.predict(X_test)

# Saving the model
print("\nSavning Model")
save_model(best_clf_ada, "ada_rf_model.sav")

# Save submit results
print("\nSaving submit results")
X_test_org = read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
save_data(base_submission, sett.RESULT_DATA_PATH, "ada_rf_results.csv",
          index=False, header=True)


"""
Ada Boost Classifier with SVC - RandomizedSearchCV results
Best Score:	 0.6157303370786517
Best Parameters:	 {'n_estimators': 500, 'learning_rate': 0.1, 'algorithm': 'SAMME'}
"""

"""
Ada Boost Classifier with RandomForestClassifier - RandomizedSearchCV results
Best Score:	 0.8314606741573034
Best Parameters:	 {'n_estimators': 300, 'learning_rate': 0.01, 'algorithm': 'SAMME'}

Ada Boost Classifier with RandomForestClassifier - GridSearch results
Best Score:	 0.8348314606741573
Best Parameters:	 {'algorithm': 'SAMME', 'learning_rate': 0.14, 'n_estimators': 290}
Kaggle score: 0.74641

"""




