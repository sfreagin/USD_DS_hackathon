# Imports
# DO NOT CHANGE THESE LINES.
import os
import json
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from feature_engine.encoding import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from joblib import dump

warnings.filterwarnings('ignore')

# DO NOT CHANGE THESE LINES.
ROOT_DIR = os.path.dirname(os.getcwd())
MODEL_INPUTS_OUTPUTS = os.path.join(ROOT_DIR, 'model_inputs_outputs/')
INPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "inputs")
INPUT_SCHEMA_DIR = os.path.join(INPUT_DIR, "schema")
DATA_DIR = os.path.join(INPUT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "training")
TEST_DIR = os.path.join(DATA_DIR, "testing")
MODEL_PATH = os.path.join(MODEL_INPUTS_OUTPUTS, "model")
MODEL_ARTIFACTS_PATH = os.path.join(MODEL_PATH, "artifacts")
OHE_ENCODER_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'ohe.joblib')
LABEL_ENCODER_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'label_encoder.joblib')
PREDICTOR_DIR_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "predictor")
PREDICTOR_FILE_PATH = os.path.join(PREDICTOR_DIR_PATH, "predictor.joblib")
IMPUTATION_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'imputation.joblib')
if not os.path.exists(MODEL_ARTIFACTS_PATH):
    os.makedirs(MODEL_ARTIFACTS_PATH)
if not os.path.exists(PREDICTOR_DIR_PATH):
    os.makedirs(PREDICTOR_DIR_PATH)

"""
Reading the schema
The schema contains metadata about the datasets.
We will use the schema to get information about the type of each feature (NUMERIC or CATEGORICAL), the id and target features.
This will be helpful in preprocessing stage.
"""
file_name = [f for f in os.listdir(INPUT_SCHEMA_DIR) if f.endswith('json')][0]
schema_path = os.path.join(INPUT_SCHEMA_DIR, file_name)
with open(schema_path, "r", encoding="utf-8") as file:
    schema = json.load(file)
features = schema['features']

numeric_features = []
categorical_features = []
for f in features:
    if f['dataType'] == 'CATEGORICAL':
        categorical_features.append(f['name'])
    else:
        numeric_features.append(f['name'])

id_feature = schema['id']['name']
target_feature = schema['target']['name']

# Reading training data
file_name = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.csv')][0]
file_path = os.path.join(TRAIN_DIR, file_name)
df = pd.read_csv(file_path)

"""
Data Preprocessing
Data preprocessing is very important before training the model, as the data may contain missing values in some cells. 
Moreover, most of the learning algorithms cannot work with categorical data, thus the data has to be encoded.
In this section we will impute the missing values and encode the categorical features. 
Afterwards the data will be ready to train the model.

In this section we imputed the missing values using the mode. However in many cases the mode is not an optimal value. 
You can try your own imputation and compare the results.
"""

# Imputing missing data
columns_with_missing_values = df.columns[df.isna().any()]
imputation_values = {}
for column in columns_with_missing_values:
    mode = df[column].mode()[0]
    df[column].fillna(mode, inplace=True)
    imputation_values[column] = mode

path = dump(imputation_values, IMPUTATION_FILE)

# Comment the above code and write you own imputation code here

# BEGIN

# CODE HERE

# END


"""
Encoding Categorical features. This step is necessary since ML models operate with numbers, not strings.
Notice that we do not want to encode the target feature nor the id column.
"""

# Saving the id and target columns in a different variable.
ids = df[id_feature]
target = df[target_feature]

# Dropping the id and target from the dataframe
df.drop(columns=[id_feature, target_feature], inplace=True)

# Ensure that all categorical columns are stored as str type.
# This is to ensure that even if the categories are numbers (1, 2, ...), they still get encoded.
for c in categorical_features:
    df[c] = df[c].astype(str)

# Encoding the features
encoder = OneHotEncoder(top_categories=6)
encoder.fit(df)
df = encoder.transform(df)

# Saving the encoder to use it on the testing dataset
dump(encoder, OHE_ENCODER_FILE)

"""Encoding the target feature"""

encoder = LabelEncoder()
y = encoder.fit_transform(target.values.reshape(-1, 1))
dump(encoder, LABEL_ENCODER_FILE)

"""
Training the Classifier
We choose Logistic Regression Classifier, but feel free to try your own and compare the results.
"""

# Creating a logistic regression model and training it
model = LogisticRegression()
model.fit(df, y)

"""Put your own model here."""

# BEGIN

# model = ...

# END


# Saving the model to use it for predictions
dump(model, PREDICTOR_FILE_PATH)
