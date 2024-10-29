import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load the breast cancer data into a Pandas dataframe and create variables for the features and target.
# Load the breast cancer data set
breast_cancer_data = load_breast_cancer()

# Create a Pandas DataFrame from the data
df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
df['target'] = breast_cancer_data.target

# Create feature and target variables
X = df.drop('target', axis=1)
y = df['target']

#view first 5 rows of data
print(df.head())

#how frequently does the positive target occur. '1' occurs 63% of time and '0' occurs 37% of time
print(df['target'].value_counts(normalize=True))

# Generate summary statistics for the data
pd.set_option('display.max_columns', None)
pd.set_option('float_format', '{:f}'.format)
print(df.describe())

# Create a pairplot for the first few features
sns.pairplot(df[['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'target']], hue='target')
plt.show()

# Create a correlation coefficient heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

#Create a boxplot for mean radius by target type
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='mean radius', data=df)
plt.title('Boxplot of Mean Radius by Target Type')
plt.xlabel('Target')
plt.ylabel('Mean Radius')
plt.show()

#Create a boxplot for worst concave points by target type
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='worst concave points', data=df)
plt.title('Boxplot of worst concave points by Target Type')
plt.xlabel('Target')
plt.ylabel('worst concave points')
plt.show()

#Create a boxplot for worst perimeter by target type
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='worst perimeter', data=df)
plt.title('Boxplot of worst perimeter by Target Type')
plt.xlabel('Target')
plt.ylabel('worst perimeter')
plt.show()

#Create a boxplot for mean fractal dimension by target type
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='worst perimeter', data=df)
plt.title('Boxplot of mean fractal dimension by Target Type')
plt.xlabel('Target')
plt.ylabel('mean fractal dimension')
plt.show()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Build and train logistic regression model
log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

#print(y_pred)
# Evaluate the model
#print(confusion_matrix(y_test, y_pred))

# Visual Representation of Confusion Matrix
# Generate a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Number of false positives and false negatives
FP = cm[0, 1]  # False Positives (predicted 1, but actual 0)
FN = cm[1, 0]  # False Negatives (predicted 0, but actual 1)
print(f"Number of False Positives: {FP}")
print(f"Number of False Negatives: {FN}")

#Use classification_report to generate further analysis of your model's predictions.
#Make sure you understand everything in the report and are able to explain what all the metrics mean.
print(classification_report(y_test, y_pred))

# Extract coefficients
coefficients = log_reg.coef_[0]
#feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
#feature_importance = feature_importance.reindex(feature_importance['Coefficient'].abs().sort_values(ascending=False).index)
#plt.figure(figsize=(10, 8))
#plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
#lt.xlabel('Coefficient')
#plt.ylabel('Feature')
#plt.title('Feature Importance in Logistic Regression')
#plt.show()

#Normalize coefficients by feature standard deviations
feature_stds = X_train.std()
normalized_coefficients = coefficients / feature_stds
#Combine feature names and coefficients into a DataFrame
feature_importance_normalized = pd.DataFrame({'Feature': X.columns, 'Coefficient': normalized_coefficients})
#Sort by absolute value of coefficients
feature_importance_normalized = feature_importance_normalized.reindex(feature_importance_normalized['Coefficient'].abs().sort_values(ascending=False).index)
#Visualize feature importances
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_normalized['Feature'], feature_importance_normalized['Coefficient'])
plt.xlabel('Normalized Coefficient')
plt.ylabel('Feature')
plt.title('Feature Importance in Logistic Regression (Normalized)')
plt.show()
#the most important predictor of cancer in this dataset is texture error