"""
Contain classes which will be used for feature transformations.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from helper import get_title
import numpy as np
import pandas as pd
import re

class NumericalAttributeAdder(BaseEstimator, TransformerMixin):
    """A class to add numerical features and transform data"""
    def __init__(self):
        self.Number_of_cabins = 0
        self.Norm_fare = 0
        self.Cabin_type = ''
        self.Family_Size = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Add number of cabins a passenger had
        X['Number_of_cabins'] = X.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

        # Normalize fare values (right skewed variable)
        X['Norm_fare'] = np.log(X.Fare + 1)

        # Family size
        X['Family_size'] = X['SibSp'] + X['Parch'] + 1

        # Drop columns
        drop_columns = ['Fare']
        X.drop(drop_columns, axis=1, inplace=True)

        return X


class CategoricalAttributeAdder(BaseEstimator, TransformerMixin):
    """A class to add categorical features and transform data"""
    def __init__(self):
        self.Cabin_type = ''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Add cabin type based on the first letter of the cabin name
        X['Cabin_type'] = X.Cabin.apply(lambda x: str(x)[0])

        # Name title
        X['Title'] = X['Name'].apply(get_title)

        # Group all non-common titles into one single grouping "Rare"
        X['Title'] = X['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        X['Title'] = X['Title'].replace('Mlle', 'Miss')
        X['Title'] = X['Title'].replace('Ms', 'Miss')
        X['Title'] = X['Title'].replace('Mme', 'Mrs')

        # Drop columns
        drop_columns = ['PassengerId', 'Ticket', 'Name', 'Cabin']
        X.drop(drop_columns, axis=1, inplace=True)

        return X

class FeatureAdder(BaseEstimator, TransformerMixin):
    """A class to add features and transform data"""
    def __init__(self):
        super.__init__();

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Available fields 'PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

        # Name title
        X['Title'] = X['Name'].apply(get_title)
        # Furthe simplify Title into fewer categories 'Mr', 'Mrs', 'Miss', 'Rare'
        # Group all non-common titles into one single grouping "Rare"
        X['Title'] = X['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        X['Title'] = X['Title'].replace('Mlle', 'Miss')
        X['Title'] = X['Title'].replace('Ms', 'Miss')
        X['Title'] = X['Title'].replace('Mme', 'Mrs')

        # Age - categorized into 4 groups base on quartile ranges
        X['Age_bin'] = pd.cut(X['Age'], bins=[0, 12, 20, 40, 120], labels=['Children', 'Teenage', 'Adult', 'Elder'])

        # Family size
        X['Family_size'] = X['SibSp'] + X['Parch'] + 1

        # Fare - categorized into 4 groups base on quartile ranges
        X['Fare_bin'] = pd.cut(X['Fare'], bins=[0, 7.91, 14.45, 31, 120], labels=['Low_fare', 'median_fare',
                                                                                  'Average_fare', 'high_fare'])

        # Add number of cabins a passenger had
        X['Number_of_cabins'] = X.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

        # Add cabin type based on the first letter of the cabin name
        X['Cabin_type'] = X.Cabin.apply(lambda x: str(x)[0])

        return X
