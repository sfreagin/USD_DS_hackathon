# Imports
# DO NOT CHANGE THESE LINES.
import os
import pandas as pd
import json
import warnings
from joblib import load
warnings.filterwarnings('ignore')


# Paths
# DO NOT CHANGE THESE LINES.
ROOT_DIR = os.path.dirname(os.getcwd())
MODEL_INPUTS_OUTPUTS = os.path.join(ROOT_DIR, 'model_inputs_outputs/')
INPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "inputs")
OUTPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "outputs")
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
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, 'predictions.csv')
if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)


# Reading the schema

file_name = [f for f in os.listdir(INPUT_SCHEMA_DIR) if f.endswith('.json')][0]
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


# Reading test data.
file_name = [f for f in os.listdir(TEST_DIR) if f.endswith('.csv')][0]
file_path = os.path.join(TEST_DIR, file_name)
df = pd.read_csv(file_path)
df.head()


"""
Data preprocessing
Note that when we work with testing data, we have to impute using the same values learned during training. 
This is to avoid data leakage.
"""

columns_with_missing_values = df.columns[df.isna().any()]
imputation_values = load(IMPUTATION_FILE)
for column in columns_with_missing_values:
    df[column].fillna(imputation_values[column], inplace=True)

"""
Encoding
We encode the data using the same encoder that we saved during training.
"""

# Saving the id column in a different variable.
ids = df[id_feature]

# Dropping the id from the dataframe
df.drop(columns=[id_feature], inplace=True)

# Encoding the rest of the features
encoder = load(OHE_ENCODER_FILE)
df = encoder.transform(df)


"""
Making predictions Using the model saved during training. 
Notice that the model outputs a 2D array with many rows and 3 columns.
Each row in the array represents an answer to a sample in the test data. 
Each number of the 3 numbers in the row is a probability to one of the 3 classes in the original problem.
"""

model = load(PREDICTOR_FILE_PATH)
# Making predictions
predictions = model.predict_proba(df)

"""
Getting the original labels.
To get the original labels back, we use the same encoder from the training phase.
Instead of calling the transform() function, this time we use inverse_transform(). 
This will convert the labels [1, 2, 3]  to the original labels [drunk_driver_involved, other, speeding_driver_involved] 
"""

# Loading the encoder from the training process
encoder = load(LABEL_ENCODER_FILE)

# Getting the original class names
class_names = encoder.inverse_transform([0, 1, 2])

# Creating the predictions dataframe
predictions = pd.DataFrame(predictions, columns=class_names)

# Inserting the id column
predictions.insert(0, 'u_id', ids)

# Saving predictions
predictions.to_csv(PREDICTIONS_FILE)
