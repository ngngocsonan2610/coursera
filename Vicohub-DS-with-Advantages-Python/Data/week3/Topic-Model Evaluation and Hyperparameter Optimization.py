import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read Housing Data
OrigData = pd.read_csv('../InputData/BostonHousingData/housing.csv')

features = OrigData.loc[:,['RM','LSTAT','PTRATIO']]
kmeans = KMeans(n_clusters=3)
model = kmeans.fit(features)
classes = model.predict(features)
OrigData['CLASS'] = ['Class ' + str(x+1) for x in classes]
print(OrigData.loc[1:20].to_string())

housing_features = OrigData.drop(['MEDV','CLASS'], axis = 1)
housing_prices = OrigData['MEDV']
housing_class = OrigData['CLASS']

print("Housing Data \n")
print(housing_features.to_string())
print(housing_prices.to_string())
print(housing_class.to_string())

# Some Polynominal functions to check
import numpy as np
X = housing_features['RM'].values
y = housing_prices.values
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    coeffs = np.polyfit(X,y,degree)
    xPoly = np.arange(min(X)-1, max(X)+1, .01)
    yPoly = np.polyval(coeffs, xPoly)
    plt.plot(xPoly, yPoly, label='degree={0}'.format(degree))
plt.xlim(3, 9)
plt.ylim(200000, 1000000)
plt.legend()
# plt.show()

# Find the optimization degree of polynimonal
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)
model = make_pipeline(PolynomialFeatures(), LinearRegression())
train_score, val_score = validation_curve(model, X[:, np.newaxis], y,
param_name='polynomialfeatures__degree',param_range=degree)
plt.figure()
plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xticks(range(21))
plt.xlabel('degree')
plt.ylabel('score')
plt.grid()
# plt.show()

# From the validation curve, we can read-off that the optimal trade-off between bias and variance is found for a
## tenth-order polynomial; we can compute and display this fit over the original data as follows:
plt.figure()
plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
degree = 10
coeffs = np.polyfit(X,y,degree)
#use more points for a smoother plot
xPoly = np.arange(min(X)-1, max(X)+1, .01)
#Evaluates the polynomial for each xPoly value
yPoly = np.polyval(coeffs, xPoly)
plt.plot(xPoly, yPoly, label='degree={0}'.format(degree))
plt.xlim(3, 9)
plt.ylim(200000, 1000000)
plt.legend()
# plt.show()

# Learning Curve
from sklearn.model_selection import learning_curve
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for i, degree in enumerate([2, 10]):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    N, train_lc, val_lc = learning_curve(model, X[:, np.newaxis], y, cv=7)
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1],
    color='gray', linestyle='dashed')
    # ax[i].set_ylim(0, 1)
    # ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')
# plt.show()

targets = 'CLASS'
data = pd.concat([housing_features, housing_class], axis=1, sort=False)
training = data.sample(frac=0.7, random_state=1)
testing = data.loc[~data.index.isin(training.index)]
TrainData = training.drop(targets, 1)
TargetTrainData = training[targets]
TestData = testing.drop(targets, 1)
TargetTestData = testing[targets]

# Grid Search
# Set the parameters by cross-validation
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#
# scores = ['precision', 'recall']
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                        scoring='%s_macro' % score)
#     clf.fit(TrainData, TargetTrainData)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = TargetTestData, clf.predict(TestData)
#     print(classification_report(y_true, y_pred))
#     print()

# Random Search
# import numpy as np
#
# from time import time
# from scipy.stats import randint as sp_randint
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.datasets import load_digits
# from sklearn.ensemble import RandomForestClassifier
#
# # get some data
# digits = load_digits()
# X, y = digits.data, digits.target
#
# # build a classifier
# clf = RandomForestClassifier(n_estimators=20)
#
#
# # Utility function to report best scores
# def report(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")
#
#
# # specify parameters and distributions to sample from
# param_dist = {"max_depth": [3, None],
#               "max_features": sp_randint(1, 11),
#               "min_samples_split": sp_randint(2, 11),
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}
#
# # run randomized search
# n_iter_search = 20
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search, cv=5)
#
# start = time()
# random_search.fit(X, y)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
# report(random_search.cv_results_)
#
# # use a full grid over all parameters
# param_grid = {"max_depth": [3, None],
#               "max_features": [1, 3, 10],
#               "min_samples_split": [2, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}
#
# # run grid search
# grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
# start = time()
# grid_search.fit(X, y)
#
# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)


