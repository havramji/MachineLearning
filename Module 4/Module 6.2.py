import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

#Load the Adult data into a Pandas Dataframe.
df = pd.read_csv(DATA_PATH, header=None)

#Ensure the dataset has properly named columns. If the columns are not read in, assign them by referencing the dataset documentation.
df.columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

#Display the first five rows of the dataset.
print(df.head())

#Do exploratory data analysis to give you some better intuition for the dataset.
#This is a bit open-ended. How many rows/columns are there?
#How are NULL values represented? What's the percentage of positive cases in the dataset?
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\nNULL value representation:")
for col in df.columns:
    print(f"Column '{col}': {df[col].unique()}")

positive_cases = df[df["income"] == " >50K"].shape[0]
total_cases = df.shape[0]
positive_percentage = (positive_cases / total_cases) * 100
print(f"\nPercentage of positive cases: {positive_percentage:.2f}%")

print("Columns with NULL values and their counts:")
print(df.isnull().sum())
# Drop rows with NULL values
df_no_nulls = df.dropna()

df.replace(' ?', np.nan, inplace=True)
df.dropna(inplace=True)

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

#Use Scikit-Learn's LabelEncoder to convert the income column with a data type string to a binary variable.
le = LabelEncoder()
df_no_nulls['income'] = le.fit_transform(df_no_nulls['income'])

# Split the data into training and test sets
X = df_no_nulls.drop('income', axis=1)
y = df_no_nulls['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

#Use Scikit-Learn's roc_auc_score to calculate the AUC score for a method that always predicts the majority class.
majority_class = y_train.mode()[0]
y_pred_majority = np.full(len(y_test), majority_class)
auc_score_majority = roc_auc_score(y_test, y_pred_majority)
print(f"AUC score for always predicting the majority class: {auc_score_majority}")

#Use Scikit-Learn's ColumnTransformer to apply One Hot Encoding to the categorical variables in workclass, education,
#marital-status, occupation, relationship, 'race', sex, and native-country. Also, apply MinMaxScaler to the remaining
#continuous features. How many columns will the dataframe have after these columns transformations are applied?
categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
numerical_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = MinMaxScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ])
X_train_transformed = preprocessor.fit_transform(X_train)
num_columns_after_transformation = X_train_transformed.shape[1]
print(f"Number of columns after transformation: {num_columns_after_transformation}")

#Create your own model in Keras to predict income in the Adult training data.
#Remember, it's always better to start simple and add complexity to the model if necessary.
#What's a good loss function to use?
def create_model():
 model = keras.Sequential()
 model.add(keras.Input(shape=(num_columns_after_transformation,)))
 model.add(Dense(64, activation='relu'))
 model.add(Dense(1, activation='sigmoid'))
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 return model

# Create the KerasClassifier
keras_clf = KerasClassifier(model=create_model, epochs=10, batch_size=32, verbose=0)

#Keras can be integrated with Scitkit-Learn using a wrapper.
#Use the KerasClassifier wrapper to integrate your Keras model with the ColumnTransformer from previous steps using a Pipeline object.
pipeline = Pipeline([
   ('preprocessor', preprocessor),
    ('classifier', keras_clf)
])

#Fit your model.
pipeline.fit(X_train, y_train)

#Calculate the AUC score of your model on the test data. Does the model predict better than random?
#The AUC score for the majority class classifier was ~0.5.
#Keras model's AUC (0.91) is much greater than 0.5 so it is better in making predictions than random model
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC score for the Keras model: {auc_score}")

#Generate an ROC curve for your model using RocCurveDisplay.
#What would the curve look like if all your predictions were randomly generated?
#A randomly generated model would produce an ROC curve that closely resembles a
#diagonal line from the bottom-left to the top-right corner of the plot.  This
#indicates that the model's predictions are no better than random chance. The AUC
#score for such a model would be approximately 0.5.

#What would the curve look like if you had a perfect model?
#A perfect model would produce an ROC curve that hugs the top-left corner of the
#plot. This means that the model achieves a true positive rate of 1 while
#maintaining a false positive rate of 0.  The AUC score for a perfect model
#would be 1.
RocCurveDisplay.from_predictions(y_test, y_pred_prob)