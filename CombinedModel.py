import csv
import math
import itertools
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from subprocess import check_output
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error

print(check_output(["ls", "/Users/Hermione/MasterUCL/Web economics/Assignment/dataset"]).decode("utf8"))


train = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/train.csv")
validation = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/validation.csv")

ytrain = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/train.csv")["click"]
yvalidation = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/validation.csv")["click"]

# this correct the missing columns in validation and test data set due to I encoded them seperately
def MissingColumnsCorrector(df1,df2):
    for columns in df1:
        if columns in df2:
            continue
        else:
            missing_columns = columns
            ind = df1.columns.get_loc(missing_columns)
            df2.insert(ind,missing_columns,0.0)
    return df2


xtrain = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/train_data.csv")

xvalidation = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/val_data.csv")


xvalidation = MissingColumnsCorrector(xtrain,xvalidation)

xtrain = np.array(xtrain)
ytrian = np.array(ytrain)
ytrain = [int(numeric_string) for numeric_string in ytrain]
xvalidation = np.array(xvalidation)
yvalidation = np.array(yvalidation)
yvalidation = [int(numeric_string) for numeric_string in yvalidation]


gbdt = GradientBoostingClassifier()
gbdt.fit(xtrain, ytrain)
feature_importance = gbdt.feature_importances_
print(feature_importance)

print(gbdt.feature_importances_.shape)


index = np.where(feature_importance > 0)
index = list(index[0])
print(len(index))

import csv

with open('importantFeatureIndCombined', 'w') as myfile:
    wr = csv.writer(myfile,  dialect='excel')
    wr.writerow(index)

    
    
