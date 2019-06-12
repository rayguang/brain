# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


# Other Libraries
#from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold

import warnings
warnings.filterwarnings("ignore")


# Load dataset
df = pd.read_csv("fraud_prep.csv")

# First 5 features
df.head()

# Dataset features
print(df.info())

# Now use V1 < -3 and V2 > 2 as threshold to identify the fraud transactions
df['fraud_flagged'] = np.where(np.logical_and(df['V1']<-3, df['V2']>2), 1,0)
ct=pd.crosstab(df.Class, df.fraud_flagged, rownames=["Fraud_actual"], colnames=["Fraud_flagged"])

print("**** Fraud_actual vs Fraud_flagged ****")
print(ct)

print("**** Fraud_actual vs Fraud_flagged (%) ****")
print(ct.apply(lambda r: r/r.sum()*100, axis=1))

# Drop columns/features
X = df.drop(["Time", "Class", "fraud_flagged"], axis=1)
y = df["Class"]

# Create training and test sets using random split
# Note that, we can also use CV, shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Define the resampling method
resample_method = SMOTE()
X_resampled, y_resampled = resample_method.fit_sample(X_train, y_train)

# check before and after resample
print("Before resampling:\n{}\n".format(y_train.value_counts()))
print("After resampling:\n{}\n".format(pd.Series(y_resampled).value_counts()))


# Define the parameter sets to test
param_grid = {"n_estimators": [10, 50], 
              "max_features": ["auto", "log2"],  
#               "min_samples_leaf": [1, 10],
              "max_depth": [4, 8], 
              "criterion": ["gini", "entropy"], 
              "class_weight": [None, {0:1, 1:12}]
}

# define the model to use
model = RandomForestClassifier(random_state=0)

# combine the parameter sets with the defined model
CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="recall", n_jobs=-1)

# fit the model to our training data and obtain best parameters
CV_model.fit(X_train, y_train)

# obtain predictions from the test data 
predicted = CV_model.predict(X_test)

# predict probabilities
probs = CV_model.predict_proba(X_test)

# print the accuracy score, ROC score, classification report and confusion matrix
print("Accuracy Score: {}\n".format(accuracy_score(y_test, predicted)))
print("ROC score = {}\n".format(roc_auc_score(y_test, probs[:,1])))
print("Classification Report:\n{}\n".format(classification_report(y_test, predicted)))
print("Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, predicted)))