from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression

import pandas as pd
import numpy as np


# using Univariate feature selection
# for more algos, visit:
# https://scikit-learn.org/stable/modules/feature_selection.html
# 1) removing feature using VarianceThreshold, removal features that variance does not meet certain threshold. 
# By default it removes all zero-variance features, i.e., feature has the same value for all samples
# 2) univariate feature selection
#    -- SelectKBest: remove all features but the K highest scoring features 

# 3)

# how many features we want to keep
KEEP_FEATURES = 2

# read csv file
data = pd.read_csv('crime_prep.csv')

# change data to array
array = data.values

# change NaN to 0
where_are_Nans = pd.isnull(array)
array[where_are_Nans] = 0

# get target from data
target = array[:,0]

# change array to list(in order to fit)
target = target.tolist()

# get features from data
features=array[:,1:]

# delete cities column from features
# 1. string cannot fit directly
# 2. only 1 feature in 126 features, not affect much
# 3. if we really want this column, we can add a function to change cities to float value
features=np.delete(features, 3, 1)

# fit, we can choose f_regression OR mutual_info_regression, we can test both and see the result
features_new = SelectKBest(f_regression, k=KEEP_FEATURES).fit_transform(features, target)

# print left features
print(features_new)
