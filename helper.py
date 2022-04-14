"""
Contain common functionalities which may commonly used in all modules
"""
import pandas as pd
import os
from settings import Settings
import re
import pickle

def load_inputs():
    """
    Read data from input csv files.
    Drop cabin "T" record from the training set, since no such records available in test set
    otherwise OneHotEncoder will results different number of fields for train set and test set
    """
    sett = Settings()
    # Read input csv files
    print("Initiate data loading process...")
    print("\tRead csv files.")
    X_train = pd.read_csv(os.path.join(sett.INPUT_DATA_PATH, sett.TRAIN_SET_FILENAME))
    X_test = pd.read_csv(os.path.join(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME))

    print("\tDrop cabin 'T' record from the training set.")
    X_train = X_train[X_train.Cabin != 'T']

    print("\tSeperate lable vector.")
    y_train = X_train['Survived']
    X_train = X_train.drop('Survived', axis=1)

    print("Data loading completed.")
    return X_train, y_train, X_test

def clean_data(train_set, test_set):
    """
    Fill values in both training set and test set.
    """
    print("Initiate cleaning process...")

    # Fill missing "Age" values with train set mean age value.
    print("\tFill empty 'Age' values with train set mean value.")
    age_mean = train_set['Age'].mean()
    print("\tMean age value: ", age_mean)
    train_set['Age'].fillna(age_mean, inplace=True)
    test_set['Age'].fillna(age_mean, inplace=True)

    # Fill missing "Embarked" values with train set mode embarked value.
    print("\tFill empty 'Embarked' values with train set mode value.")
    train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)

    # Fill missing "Fare" value in the test set with the fare mode value of the training set
    print("\tFill empty 'Fare' values with train set mode fare value.")
    test_set['Fare'].fillna(train_set['Fare'].mode()[0], inplace=True)

    # Fill null cabin values with letter 'N'
    print("\tFill empty 'Cabin' values with letter 'N'")
    train_set.Cabin.fillna("N", inplace=True)
    test_set.Cabin.fillna("N", inplace=True)

    print("Data cleaning process completed.")

    return train_set, test_set

def organize_features(data_set):
    """
    Available fields 'PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
    Add new fields.
    Drop unwanted fields.
    """
    print("Initiate adding features...")
    # Name title
    print("\tAdd 'Title'")
    data_set['Title'] = data_set['Name'].apply(get_title)
    # Further simplify Title into fewer categories 'Mr', 'Mrs', 'Miss', 'Rare'
    # Group all non-common titles into one single grouping "Rare"
    data_set['Title'] = data_set['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data_set['Title'] = data_set['Title'].replace('Mlle', 'Miss')
    data_set['Title'] = data_set['Title'].replace('Ms', 'Miss')
    data_set['Title'] = data_set['Title'].replace('Mme', 'Mrs')

    # Age - categorized into 4 groups base on quartile ranges
    print("\tAdd 'Age_bin'")
    data_set['Age_bin'] = pd.cut(data_set['Age'], bins=[0, 12, 20, 40, 120],
                                 labels=['Children', 'Teenage', 'Adult', 'Elder'])

    # Family size
    print("\tAdd 'Family_size'")
    data_set['Family_size'] = data_set['SibSp'] + data_set['Parch'] + 1

    # Fare - categorized into 4 groups base on quartile ranges
    print("\tAdd 'Fare_bin'")
    data_set['Fare_bin'] = pd.cut(data_set['Fare'], bins=[0, 7.91, 14.45, 31, 120],
                                  labels=['Low_fare', 'median_fare', 'Average_fare', 'high_fare'])

    # Add number of cabins a passenger had
    print("\tAdd 'Number_of_cabins'")
    data_set['Number_of_cabins'] = data_set.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

    # Add cabin type based on the first letter of the cabin name
    print("\tAdd 'Cabin_type'")
    data_set['Cabin_type'] = data_set.Cabin.apply(lambda x: str(x)[0])

    # Drop columns
    print("\tDrop columns 'PassengerId', 'Ticket', 'Name', 'Cabin'")
    drop_columns = ['PassengerId', 'Ticket', 'Name', 'Cabin', 'Age', 'Fare']
    data_set.drop(drop_columns, axis=1, inplace=True)
    
    print("Feature adding completed.")

    return data_set

def get_title(name):
    """
    Define function to extract titles from passenger names
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def save_data(data_set, path, filename, index=False, header=False):
    """Write to a csv"""
    os.makedirs(path, exist_ok=True)
    file = pd.DataFrame(data=data_set)
    csv_path = os.path.join(path, filename)
    file.to_csv(path_or_buf=csv_path, index=index, header=header)

def read_data(path, file_name):
    """Read input csv files"""
    print("\tRead csv files.")
    dataset = pd.read_csv(os.path.join(path, file_name))
    return dataset

def save_model(model, model_name):
    """Save model"""
    print("\tSave model: ", model_name)
    sett = Settings()
    try:
        os.mkdir(sett.SAVE_MODEL_PATH)
        print("\t", sett.SAVE_MODEL_PATH)
    except OSError as error:
        print("\t", sett.SAVE_MODEL_PATH)

    file_name = os.path.join(sett.SAVE_MODEL_PATH, model_name)
    pickle.dump(model, open(file_name, 'wb'))

def load_model(model_name):
    """Load model"""
    print("\tLoad model: ", model_name)
    sett = Settings()
    file_name = os.path.join(sett.SAVE_MODEL_PATH, model_name)
    loaded_model = pickle.load(open(file_name, 'rb'))

    return loaded_model

def show_results(cv):
    print("Scross validation scores: ", cv)
    print("Scross validation mean: ", cv.mean())

