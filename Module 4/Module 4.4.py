import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Use load_breast_cancer to load the Breast Cancer Wisconsin dataset as a Pandas dataframe.
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

#Split the dataset into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.2, random_state=32
)

# Display the first five rows of data and make sure everything looks ok.
# You should have already explored the data a bit in the logistic regression mini-project so there's no need to conduct further EDA.
print(df.head())

#Use Scikit-Learn's DecisionTreeClassifier to fit a model on the training data.
clf = DecisionTreeClassifier(random_state=32)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

#Visualize the resulting tree using plot_tree.
plt.figure(figsize=(30, 20))
plot_tree(clf, feature_names=data.feature_names, class_names=['0', '1'], filled=True)
plt.show()

#Decision Tree changes when random_state changes.  Accuracy of Decision Tree is 87.7%

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for the tree is: {accuracy}")

#Iterate on the first two steps by trying different inputs to the decision tree classifier.
#What happens if you change the max depth? How about the maximum number of leaf nodes?
#Accuracy of model increases as max_depth increases. The accuracy reaches a peak around 93.8 for higher max_depth and then plateaus. Further increases to max_depth/max_leaf_nodes does not make a difference
#gini (degree of probability of a particular variable being wrongly classified when it is randomly chosen) improves at every depth
#Lower max_leaf_nodes reduces max depth even though more max depth is requested in the model
#From the visualization, make sure you're able to understand how to descend the decision tree to arrive at a prediction.
for max_depth in [2, 5, 7, 10, 15]:
    for max_leaf_nodes in [5, 10, 15, 20]:
        clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=32)
        clf.fit(X_train, y_train)

        plt.figure(figsize=(20, 10))
        plot_tree(clf, feature_names=data.feature_names, class_names=['0', '1'], filled=True)
        plt.title(f"Decision Tree with max_depth={max_depth}, max_leaf_nodes={max_leaf_nodes}")
        plt.show()

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for max_depth={max_depth}, max_leaf_nodes={max_leaf_nodes}: {accuracy}")

#Use your training data to train a Random Forest using RandomForestClassifier.
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

#Extract the feature importances from the trained model.
feature_importances = rf_clf.feature_importances_
importances_df = pd.DataFrame({'Feature': data.feature_names, 'Importance': feature_importances})

#Print the feature importances from largest to smallest.
#Top 5 features important for model prediction are worst area, worst concave points, worst radius, worst_perimeter and mean perimeter
importances_df = importances_df.sort_values(by='Importance', ascending=False)
print(importances_df)

#Build and train an AdaBoostClassifier on your training data using a decision tree of max depth equal to 1 as your weak learner.
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
ada_clf.fit(X_train, y_train)

# Train Decision Tree model
dt_classifier = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=5)  # Example hyperparameters
dt_classifier.fit(X_train, y_train)

# Train Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5) # Example hyperparameters
rf_classifier.fit(X_train, y_train)

# Train AdaBoost model
ada_classifier = AdaBoostClassifier(n_estimators=50, estimator=DecisionTreeClassifier(max_depth=1))
# ada_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
# Example hyperparameters
ada_classifier.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_classifier.predict(X_test)
rf_predictions = rf_classifier.predict(X_test)
ada_predictions = ada_classifier.predict(X_test)

# Evaluate models
dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)
ada_accuracy = accuracy_score(y_test, ada_predictions)

print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"AdaBoost Accuracy: {ada_accuracy}")

# Determine the best performing model
best_model = max([(dt_accuracy, "Decision Tree"), (rf_accuracy, "Random Forest"), (ada_accuracy, "AdaBoost")], key=lambda x: x[0])
print(f"\nThe best performing model is {best_model[1]} with an accuracy of {best_model[0]}")