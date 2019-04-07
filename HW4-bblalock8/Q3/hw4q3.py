## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
print("Loading Data")
data = pd.read_csv('eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# XXX
print("Splitting Data")
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True, random_state=random_state)


# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX
print("Training LinReg Classifier")
reg = LinearRegression().fit(X_train, y_train)

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
# XXX
print('Linear Regression Accuracy of Training Set')
y_predict = reg.predict(X_train)
print(round(accuracy_score(y_train, y_predict.round(0)), 2))

print('Linear Regression Accuracy of Test Set')
y_predict = reg.predict(X_test)
print(round(accuracy_score(y_test, y_predict.round(0)), 2))

# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX
print("Training LinReg Classifier")
rf = RandomForestClassifier().fit(X_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print('RandomForest Accuracy of Training Set')
y_predict = rf.predict(X_train)
print(round(accuracy_score(y_train, y_predict.round(0)), 2))

print('RandomForest Accuracy of Test Set')
y_predict = rf.predict(X_test)
print(round(accuracy_score(y_test, y_predict.round(0)), 2))


# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
# XXX
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

f_names = x_data.columns[indices]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_data.shape[1]):
    print("%d. feature %d: %s (%f)" % (f + 1, indices[f], f_names[f], importances[indices[f]]))

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX
parameters = {'n_estimators':range(100, 400, 100), 'max_depth':range(5, 20, 5)}
rf_tuned = GridSearchCV(RandomForestClassifier(), parameters, cv=10, n_jobs=-1)
rf_tuned = rf_tuned.fit(x_data, y_data)
print("Random Forest best params: {}".format(rf_tuned.best_params_))
print("Random Forest best score: {}".format(round(rf_tuned.best_score_, 2)))

print('RandomForest Tuned Accuracy of Test Set')
y_predict = rf_tuned.predict(X_test)
print(round(accuracy_score(y_test, y_predict.round(0)), 2))


# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


print("Training SVM Classifier")
svm = SVC(gamma='auto').fit(X_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print('SVM Accuracy of Training Set')
y_predict = svm.predict(X_train)
print(round(accuracy_score(y_train, y_predict.round(0)), 2))

print('SVM Accuracy of Test Set')
y_predict = svm.predict(X_test)
print(round(accuracy_score(y_test, y_predict.round(0)), 2))

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX
parameters = {'kernel':('linear', 'rbf'), 'C':[0.0001, 0.1, 100]}
svm_tuned = GridSearchCV(SVC(gamma='auto'), parameters, cv=10, n_jobs=-1)
svm_tuned = svm_tuned.fit(normalize(x_data), y_data)
print("SVM best params: {}".format(svm_tuned.best_params_))
print("SVM best score: {}".format(round(svm_tuned.best_score_, 2)))

print('SVM Tuned Accuracy of Test Set')
y_predict = svm_tuned.predict(X_test)
print(round(accuracy_score(y_test, y_predict.round(0)), 2))


# XXX
# TODO: Calculate the mean training score, mean testing score and mean fit time for the
# best combination of hyperparameter values that you obtained in Q3.2. The GridSearchCV
# class holds a  ‘cv_results_’ dictionary that should help you report these metrics easily.
# XXX
cv_results = svm_tuned.cv_results_
best_param_index = np.where(np.array(cv_results['params'])==svm_tuned.best_params_)[0][0]
print("SVM mean trainging score: {}"
      .format(cv_results['mean_train_score'][best_param_index])
      )

print("SVM mean test score: {}"
      .format(cv_results['mean_test_score'][best_param_index])
      )

print("SVM mean fit time score: {}"
      .format(cv_results['mean_fit_time'][best_param_index])
      )


# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.
# XXX
pca = PCA(n_components=10, svd_solver='full')
pca.fit(x_data)
print([round(val, 2) for val in pca.explained_variance_ratio_])
print([round(val, 2) for val in pca.singular_values_])

