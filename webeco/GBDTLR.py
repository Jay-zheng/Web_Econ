from sklearn.ensemble import GradientBoostingClassifier

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

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

ind = []
for ii in feature_importance.index:
    feature = feature_importance.loc(ii)
    if feature > 0:
        ind = ind.append(ii)
    else:
        continue


new_train = xtrain[:, ind]
new_val = xvalidation[:, ind]

lr1 = LogisticRegression(class_weight = "balanced")
lr1.fit(new_train, ytrain)
lr1_predict_labels = lr1.predict(new_val)

correctpred = sum(lr1_predict_labels == yvalidation)

print(correctpred)