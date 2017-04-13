import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression



from subprocess import check_output
print(check_output(["ls", "/Users/Hermione/MasterUCL/Web economics/Assignment/dataset"]).decode("utf8"))

ytrain = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/train.csv")["click"]
yvalidation = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/validation.csv")["click"]


def MissingColumnsCorrector(df1,df2):
    for columns in df1:
        if columns in df2:
            continue
        else:
            missing_columns = columns
            ind = df1.columns.get_loc(missing_columns)
            df2.insert(ind,missing_columns,0.0)
    return df2


xtrain = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/cleantrain.csv")
#xtest = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/cleantest.csv")
xvalidation = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/cleanvalidation.csv")

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

with open('importantFeatureInd', 'w') as myfile:
    wr = csv.writer(myfile,  dialect='excel')
    wr.writerow(index)


new_train = xtrain[:, index]
new_val = xvalidation[:, index]

lr1 = LogisticRegression(class_weight="balanced")
lr1.fit(new_train, ytrain)
lr1_predict_labels = lr1.predict(new_val)
prob = lr1.predict_proba(new_val)

pClick = pd.DataFrame(prob)

from sklearn.metrics import precision_score

precision = precision_score(yvalidation, lr1_predict_labels, average='weighted')

print(pClick)
print(precision)


correctpred = sum(lr1_predict_labels == yvalidation)

print(correctpred)

