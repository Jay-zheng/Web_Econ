import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


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


xtrain = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/train_data.csv")

xvalidation = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/val_data.csv")


xvalidation = MissingColumnsCorrector(xtrain,xvalidation)

xtrain = np.array(xtrain)
ytrian = np.array(ytrain)
ytrain = [int(numeric_string) for numeric_string in ytrain]
xvalidation = np.array(xvalidation)
yvalidation = np.array(yvalidation)
yvalidation = [int(numeric_string) for numeric_string in yvalidation]

index = [0,32,33,35,37,38,40,44,51,60,73,85,88,95,100,104,107,111,112,116,118,123,157,158,159,160,161,168,174,175,176,180]
print("The number of important feature is:",len(index))

new_train = xtrain[:, index]
new_val = xvalidation[:, index]


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              "penalty": ["l1", "l2"]}

lr1 = LogisticRegression(class_weight="balanced")

grid = GridSearchCV(estimator=lr1, param_grid=param_grid)
grid.fit(new_train, ytrain)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_)

