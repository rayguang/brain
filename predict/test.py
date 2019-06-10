from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression

import pandas as pd
import numpy as np

KEEP_FEATURES = 2

data = pd.read_csv('./data/crime_prep.csv')

array = data.values

where_are_Nans = pd.isnull(array)

array(where_are_Nans) = 0