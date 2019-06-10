import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
from matplotlib import pyplot as plt

import pickle
from joblib import dump,load

# using SVM
# for details, visit:
# https://scikit-learn.org/stable/modules/svm.html#regression

# ****************************************************************
# Data preprocessing
# ****************************************************************
# Same as feature_extraction.py
data = pd.read_csv('crime_prep.csv')
array = data.values
where_are_Nans = pd.isnull(array)
array[where_are_Nans] = 0
target = array[:,0]
target = target.tolist()
features = array[:,1:]
features=np.delete(features,3,1)

random_state=12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(features):
    print("%s %s" % (train[:2], test[:2]))


# ****************************************************************
# Train model
# ****************************************************************
# read SVM model
clf = svm.SVR(kernel='RBF', C=1.0, epsilon=0.2)


#clf.fit(features, target)
scores = cross_val_score(clf, features, target, cv=5)
print("cross validation score: ", scores)
print("Accuracy: %0.2f(+/- %0.2f)"%(scores.mean(), scores.std()*2))

# ****************************************************************
# Make cross-validation predictions
# ****************************************************************
predictions = cross_val_predict(clf, features, target, cv=5)
plt.scatter(target, predictions, edgecolors=(0,0,0))
plt.plot([min(target),max(target)], [min(target), max(target)], 'k--', lw=4)
plt.xlabel("Measured") 
plt.ylabel("Predicted")
plt.show()

accuracy = metrics.r2_score(target, predictions)
print("Cross-predicted Accuracy:", accuracy)
# test using the first row
#test=[8,0,0,1,0.19,0.33,0.02,0.9,0.12,0.17,0.34,0.47,0.29,0.32,0.2,1.0,0.37,0.72,0.34,0.6,0.29,0.15,0.43,0.39,0.4,0.39,0.32,0.27,0.27,0.36,0.41,0.08,0.19,0.1,0.18,0.48,0.27,0.68,0.23,0.41,0.25,0.52,0.68,0.4,0.75,0.75,0.35,0.55,0.59,0.61,0.56,0.74,0.76,0.04,0.14,0.03,0.24,0.27,0.37,0.39,0.07,0.07,0.08,0.08,0.89,0.06,0.14,0.13,0.33,0.39,0.28,0.55,0.09,0.51,0.5,0.21,0.71,0.52,0.05,0.26,0.65,0.14,0.06,0.22,0.19,0.18,0.36,0.35,0.38,0.34,0.38,0.46,0.25,0.04,0.0,0.12,0.42,0.5,0.51,0.64,0.03,0.13,0.96,0.17,0.06,0.18,0.44,0.13,0.94,0.93,0.03,0.07,0.1,0.07,0.02,0.57,0.29,0.12,0.26,0.2,0.06,0.04,0.9,0.5,0.32,0.14]
# get result 0.27
# real result is 0.2, 
#print(clf.predict([test]))


# Save model
saved_model = pickle.dumps(clf)
# Test saved model
# clf2 = pickle.loads(saved_model)
# clf2.predict()

# Save model to file
dump(clf, 'predict_svm.joblib')