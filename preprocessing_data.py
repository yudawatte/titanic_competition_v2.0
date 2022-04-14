"""
Read data from input csv files, clean data, alter features, and save
processed data.
"""
from helper import load_inputs, clean_data, organize_features, save_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from settings import Settings

# Load input data
X_train, y_train, X_test = load_inputs()
print("Read input data")
print("\tTrain set shape: ",X_train.shape)
print("\tTrain target shape",y_train.shape)
print("\tTest set shape", X_test.shape)

# Clean data
X_train, X_test = clean_data(X_train, X_test)
print("After cleaning")
print("\tTrain set shape: ",X_train.shape)
print("\tTrain target shape",y_train.shape)
print("\tTest set shape", X_test.shape)

# Add features
X_train = organize_features(X_train)
X_test = organize_features(X_test)

print("After add features")
print("\tTrain set shape: ",X_train.shape)
print("\tTrain target shape",y_train.shape)
print("\tTest set shape", X_test.shape)

# Separate numerical and categorical columns
num_columns = ['SibSp', 'Parch', 'Family_size', 'Number_of_cabins']
cat_columns = ['Pclass', 'Sex', 'Embarked', 'Title','Age_bin', 'Fare_bin', 'Cabin_type']

# Scale numerical columns with Standard Scalar
# Encode categorical columns with OneHotEncoder
num_attr_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

cat_attr_pipeline = Pipeline([
    ('cat_encode', OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ("num", num_attr_pipeline, num_columns),
    ("cat", cat_attr_pipeline, cat_columns),
])

X_train_scaled = full_pipeline.fit_transform(X_train)
X_test_scaled = full_pipeline.fit_transform(X_test)

print("After column transform")
print("\tTrain set shape: ",X_train_scaled.shape)
print("\tTrain target shape",y_train.shape)
print("\tTest set shape", X_test_scaled.shape)

# Save processed data
sett = Settings()
save_data(X_train_scaled, sett.PROCESSED_DATA_PATH, sett.PROCESSED_TRAIN_FILENAME, header=True)
save_data(X_test_scaled, sett.PROCESSED_DATA_PATH, sett.PROCESSED_TEST_FILENAME, header=True)
save_data(y_train, sett.PROCESSED_DATA_PATH, sett.TRAIN_TARGET_SET_FILENAME, header=True)