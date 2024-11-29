import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#1 Load the parquet file from the URL into a Pandas DataFrame
df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet')

#2 Print the first 5 rows of data
print(df.head())

#3 Drop any rows of data that contain NULL values.
df = df.dropna()

#4 Create a new feature, 'trip_duration' that captures the duration of the trip in minutes.
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

#5 Create a variable named 'target_variable' to store the name of the thing we're trying to predict, 'total_amount'.
target_variable = 'total_amount'

#6 Create a list called 'feature_cols' containing the feature names that we'll be using to predict our target variable.
# The list should contain 'VendorID', 'trip_distance', 'payment_type', 'PULocationID', 'DOLocationID', and 'trip_duration'.
feature_cols = ['VendorID', 'trip_distance', 'payment_type', 'PULocationID', 'DOLocationID', 'trip_duration']

#1 Use Scikit-Learn's train_test_split to split the data into training and test sets. Don't forget to set the random state.
X = df[feature_cols]
y = df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

#1 Create a model that always predicts the mean total fare of the training dataset.
# Use Scikit-Learn's mean_absolute_error to evaluate this model. Is it any good?
mean_fare = y_train.mean()
baseline_predictions = np.full(y_test.shape, mean_fare)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
print(f"Baseline MAE: {baseline_mae}")

#1 Use Scikit-Learn's ColumnTransformer to preprocess the categorical and continuous features independently.
# Apply the StandardScaler to the continuous columns and OneHotEncoder to the categorical columns.
categorical_cols = ['VendorID', 'payment_type', 'PULocationID', 'DOLocationID']
continuous_cols = ['trip_distance', 'trip_duration']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

#2 Integrate the preprocessor in the previous step with Scikit-Learn's LinearRegression model using a Pipeline.
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', LinearRegression())])

#3 Train the pipeline on the training data.
model.fit(X_train, y_train)

#4 Evaluate the model using mean absolute error as a metric on the test data. Does the model beat the baseline?
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Linear Regression MAE: {mae}")
print("Since Linear Regression MAE is lower than baseline MAE, Linear Regression is better than baseline.")
print("A lower MAE indicates that, on average, the model's predictions are closer to the actual values, making it a better fit for the data.")

#1 Build a Random Forest Regressor model using Scikit-Learn's RandomForestRegressor and train it on the train data.
rf_model = RandomForestRegressor(n_estimators=1, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print(f"Random Forest MAE: {rf_mae}")

#2 Evaluate the performance of the model on the test data using mean absolute error as a metric.
# Mess around with various input parameter configurations to see how they affect the model.
# Can you beat the performance of the linear regression model?
estimators = [1]
rf_results_list = []
for estimator in estimators:
    rf_model = RandomForestRegressor(n_estimators=estimator, random_state=40)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_results_list.append([estimator, mean_absolute_error(y_test, rf_predictions)])

# Gather everything into a DataFrame
rf_results_df = pd.DataFrame(rf_results_list, columns=['estimator', 'rf_mae'])
print(rf_results_df)

#1 Perform a grid-search on a Random Forest Regressor model.
# Only search the space for the parameters 'n_estimators', 'max_depth', and 'min_samples_split'.
# Note,this can take some time to run. Make sure you set reasonable boundaries for the search space.
# Use Scikit-Learn's GridSearchCV method.
parameter_grid = {
    'n_estimators': [1, 5, 50, 100],
    'max_depth': [3, 5, 7, 20],
    'min_samples_split': [3, 5, 10, 20]
}

rf_model = RandomForestRegressor(random_state=40)
grid_search = GridSearchCV(estimator=rf_model, param_grid=parameter_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
rf_mae = mean_absolute_error(y_test, rf_predictions)
print(f"Best Random Forest MAE: {rf_mae}")
print(f"Best Hyperparameters: {grid_search.best_params_}")
