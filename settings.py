"""
Contain data paths and namings.
"""
import os

class Settings():
    """A class to store all settings for the titanic competition"""

    def __init__(self):
        self.INPUT_DATA_PATH = os.path.join("data", "inputs")
        self.PROCESSED_DATA_PATH = os.path.join("data", "processed_data")
        self.SAVE_MODEL_PATH = os.path.join("data", "models")
        self.RESULT_DATA_PATH = os.path.join("data", "results")
        self.TRAIN_SET_FILENAME = "train.csv"
        self.TRAIN_TARGET_SET_FILENAME = "target.csv"
        self.TEST_SET_FILENAME = "test.csv"
        self.PROCESSED_TRAIN_FILENAME = "train_processed.csv"
        self.PROCESSED_TEST_FILENAME = "test_processed.csv"
        self.LR_RESULT_FILENAME = "lr_results.csv"
        self.RC_RESULT_FILENAME = "rc_results.csv"
        self.KNN_RESULT_FILENAME = "knn_results.csv"
        self.SVC_RESULT_FILENAME = "svc_results.csv"
        self.RF_RESULT_FILENAME = "rf_results.csv"
        self.GNB_RESULT_FILENAME = "gnb_results.csv"
        self.ADA_RESULT_FILENAME = "ada_results.csv"
        self.GBC_RESULT_FILENAME = "gbc_results.csv"
        self.LDA_RESULT_FILENAME = "lda_results.csv"
        self.VC_RESULT_FILENAME = "vc_results.csv"
        self.LR_MODEL_NAME = "lr_model.sav"
        self.RC_MODEL_NAME = "rc_model.sav"
        self.KNN_MODEL_NAME = "knn_model.sav"
        self.SVC_MODEL_NAME = "svc_model.sav"
        self.RF_MODEL_NAME = "rf_model.sav"
        self.GNB_MODEL_NAME = "gnb_model.sav"
        self.ADA_MODEL_NAME = "ada_model.sav"
        self.GBC_MODEL_NAME = "gbc_model.sav"
        self.LDA_MODEL_NAME = "lda_model.sav"
        self.VC_MODEL_NAME = "vc_model.sav"
