import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

import time

import warnings
warnings.filterwarnings("ignore")

# using decisionTree
# for details, visit:
# https://scikit-learn.org/stable/modules/tree.html#regression

# Same as feature_extraction.py
# data = pd.read_csv('crime_prep.csv')
# array = data.values
# where_are_Nans = pd.isnull(array)
# array[where_are_Nans] = 0
# target = array[:,0]
# target = target.tolist()
# features = array[:,1:]
# features=np.delete(features,3,1)

df_init     = pd.read_csv("./crime_prep.csv")
df          = df_init.drop(['v_cont_0', 'v_cat_0', 'v_cat_1', 'v_cat_2', 'v_cat_3'], axis=1)
impute      = Imputer(missing_values="NaN", strategy='mean', axis=0)
df_impute   = impute.fit(df)
df          = impute.transform(df_impute)
df          = df.drop(["target"], axis = 1)

features = pd.DataFrame(df)
target = pd.DataFrame(df_init, columns=["target"])

# read decisionTree model
clf = tree.DecisionTreeRegressor()
# train model
clf = clf.fit(features, target)

features_imp    = clf.feature_importances_
top_features    = 10
top_index       = features_imp.argsort()[-top_features:][::-1]
features_list   = []
feature_names   = []

for index in top_index:
    features_list.append(features_imp[index])
    feature_names.append(features_imp[index])

print("Main Features")    
print(features_list)
plt.figure()
df_list = pd.DataFrame(features_list,feature_names)
df_list.plot.bar()
plt.xlabel('Top '+str(top_features)+' Predictive Features')
plt.ylabel('Feature Importances')
plt.show()

# test using the first row
#test=[8,0,0,1,0.19,0.33,0.02,0.9,0.12,0.17,0.34,0.47,0.29,0.32,0.2,1.0,0.37,0.72,0.34,0.6,0.29,0.15,0.43,0.39,0.4,0.39,0.32,0.27,0.27,0.36,0.41,0.08,0.19,0.1,0.18,0.48,0.27,0.68,0.23,0.41,0.25,0.52,0.68,0.4,0.75,0.75,0.35,0.55,0.59,0.61,0.56,0.74,0.76,0.04,0.14,0.03,0.24,0.27,0.37,0.39,0.07,0.07,0.08,0.08,0.89,0.06,0.14,0.13,0.33,0.39,0.28,0.55,0.09,0.51,0.5,0.21,0.71,0.52,0.05,0.26,0.65,0.14,0.06,0.22,0.19,0.18,0.36,0.35,0.38,0.34,0.38,0.46,0.25,0.04,0.0,0.12,0.42,0.5,0.51,0.64,0.03,0.13,0.96,0.17,0.06,0.18,0.44,0.13,0.94,0.93,0.03,0.07,0.1,0.07,0.02,0.57,0.29,0.12,0.26,0.2,0.06,0.04,0.9,0.5,0.32,0.14]

# get exactly 0.2 result
#print(clf.predict([test]))