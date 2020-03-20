import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read Housing Data
OrigData = pd.read_csv('../InputData/MartSalesData/Train_UWu5bXk.csv')

# checking the percentage of missing values in each variable
print(OrigData.isnull().sum()/len(OrigData)*100)
OrigData['Item_Weight'] = OrigData['Item_Weight'].fillna(np.nanmedian(OrigData['Item_Weight']))
OrigData['Outlet_Size'].fillna(OrigData['Outlet_Size'].mode()[0], inplace=True)
print(OrigData.isnull().sum()/len(OrigData)*100)

# OrigData['CLASS'] = 0
# for idx, row in OrigData.iterrows():
#     if row['Item_Outlet_Sales'] >= 6000 :
#         OrigData.loc[idx,'CLASS'] = 'High'
#     elif row['Item_Outlet_Sales'] >= 2000 :
#         OrigData.loc[idx, 'CLASS'] = 'Medium'
#     else :
#         OrigData.loc[idx, 'CLASS'] = 'Low'

features = OrigData.loc[:,['Item_Weight','Item_MRP','Outlet_Establishment_Year']]
kmeans = KMeans(n_clusters=3)
model = kmeans.fit(features)
classes = model.predict(features)
OrigData['CLASS'] = ['Class ' + str(x+1) for x in classes]
print(OrigData.loc[1:20].to_string())

mart_features = OrigData.drop(['Item_Outlet_Sales','CLASS'], axis = 1)
mart_target = OrigData['Item_Outlet_Sales']
mart_class = OrigData['CLASS']

print("Mart Sale Data \n")
print(mart_features.loc[1:10].to_string())
print(mart_target.loc[1:10].to_string())
print(mart_class.loc[1:10].to_string())

# import dill
# filename = '../InputData/DigitializationMS01.pkl'
# dill.dump_session(filename)
#
# import pandas as pd
# import dill
# filename = '../InputData/DigitializationMS01.pkl'
# dill.load_session(filename)

#Show variance
var = mart_features.var()
print("variance of each feature:")
print(var)

mart_features = mart_features.drop(['Item_Visibility'], axis = 1)
mart_features = mart_features.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)

print("Mart Sale Data \n")
print(mart_features.loc[1:10].to_string())
print(mart_target.loc[1:10].to_string())
print(mart_class.loc[1:10].to_string())

# Feature Importance
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=1, max_depth=10)
mart_features=pd.get_dummies(mart_features)
model.fit(mart_features,mart_target)

features = mart_features.columns
importances = model.feature_importances_
nSelectedFeature = 10
indices = np.argsort(importances)[-(nSelectedFeature-1):]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
# plt.show()

# Show Feature Ranking
from sklearn.feature_selection import SelectFromModel
feature = SelectFromModel(model)
Fit = feature.fit_transform(mart_features,mart_target)
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import datasets
lreg = LinearRegression()
rfe = RFE(lreg, 10)
rfe = rfe.fit_transform(mart_features, mart_target)
from sklearn.feature_selection import f_regression
ffs = f_regression(mart_features,mart_target )
rankinfo = ffs[0]
print(rankinfo)

features = mart_features.columns
nSelectedFeature = 10
indices = np.argsort(rankinfo)[-(nSelectedFeature-1):]
plt.title('Feature Ranking')
plt.barh(range(len(indices)), rankinfo[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Ranking')
# plt.show()

# Dimensionality Reduction
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
#
# x = StandardScaler().fit_transform(mart_features) # Standardizing the features
# nSelectedFeature = len(mart_features.columns)
# SelectedAttList = []
# for i in range(1, nSelectedFeature + 1):
#     SelectedAttList.append("PC" + str(i))
#
# pca = PCA(n_components=nSelectedFeature)
# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data=principalComponents, columns=SelectedAttList)
# PCA_Data = principalDf
# PCA_Data['CLASS'] = mart_class
# PCA_Data = PCA_Data.dropna()
# print(PCA_Data.loc[1:10,:].to_string())


# Test Dimensionality Reduction with PCA
# targets = 'CLASS'
# # data = pd.concat([housing_features, housing_class], axis=1, sort=False)
# data = PCA_Data.copy()
# training = data.sample(frac=0.7, random_state=1)
# testing = data.loc[~data.index.isin(training.index)]
# TrainData = training.drop(targets, 1)
# TargetTrainData = training[targets]
# TestData = testing.drop(targets, 1)
# TargetTestData = testing[targets]
# # Apply the Logistic Regression
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn import metrics
# Model = LogisticRegression()
# # training by Logistic Regression
# Model.fit(TrainData, TargetTrainData.values.ravel())
# # predict the test data
# PredictTestData = Model.predict(TestData)
# # compare and calculate accuracy
# Accuracy = accuracy_score(TargetTestData, PredictTestData)
# print(Model)
# print('PCA Logistic regression accuracy: {:.3f}'.format(Accuracy))

# LDA Classification
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# model = LDA(n_components=3)
# LDAModel = model.fit(TrainData, TargetTrainData)
# # predict the test data
# PredictTestData = LDAModel.predict(TestData)
# # compare and calculate accuracy
# Accuracy = accuracy_score(TargetTestData, PredictTestData)
# print(LDAModel)
# print('LDA accuracy: {:.3f}'.format(Accuracy))


# KNN for classification
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
from sklearn.model_selection import train_test_split
# split the data with 50% in each set
X1, X2, y1, y2 = train_test_split(mart_features, mart_class, random_state=0,train_size=0.5)
from sklearn.metrics import accuracy_score
# fit the model on one set of data
model.fit(X1, y1)
# evaluate the model on the second set of data
y2_model = model.predict(X2)
accuracy01 = accuracy_score(y2, y2_model)
print("Accuracy 1:" + str(accuracy01))

# KNN with two-ford cross validation
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy02 = accuracy_score(y1, y1_model)
accuracy03 = accuracy_score(y2, y2_model)
print("Accuracy 2:" + str(accuracy02))
print("Accuracy 3:" + str(accuracy03))

# KNN with 5-ford cross validation
from sklearn.model_selection import cross_val_score
accuracy04 = cross_val_score(model, mart_features, mart_class, cv=5)
print("Accuracy 4:" + str(accuracy04))

# KNN with LeaveOneOut cross validation
# from sklearn.model_selection import LeaveOneOut
# scores = cross_val_score(model, mart_features.loc[1:1000], mart_class.loc[1:1000], cv=LeaveOneOut())
# print(scores)
# print("Score Mean:" + str(scores.mean()))
