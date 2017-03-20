import csv
import random
import numpy as np
import pandas as pd
from sklearn import linear_model, tree, lda, naive_bayes
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV



train = pd.read_csv("C:/Users/JayZheng/Documents/Data Science/web economics/assignment 1/dataset/train.csv")
validation = pd.read_csv("C:/Users/JayZheng/Documents/Data Science/web economics/assignment 1/dataset/validation.csv")
test = pd.read_csv("C:/Users/JayZheng/Documents/Data Science/web economics/assignment 1/dataset/test.csv")


df = pd.DataFrame()
df['advertiser'] = np.sort(train.advertiser.unique())
df['impressions'] = train.groupby('advertiser').size().values
df['click'] = train.groupby('advertiser').click.aggregate(np.sum).values
df['cost'] = train.groupby('advertiser').payprice.aggregate(np.sum).values
df['CTR'] = (((df.click / df.impressions) * 100).round(2)).astype(str) + '%'
df['CPM'] = (((df.cost / df.impressions) * 1000).round(2)).astype(str)
df['eCPC'] = ((df.cost / df.click).round(2)).astype(str)

print(df)


def CTR(variable):
    df = pd.DataFrame()
    df[variable] = np.sort(train[variable].unique())
    df['impressions'] = train.groupby(variable).size().values
    df['click'] = train.groupby(variable).click.aggregate(np.sum).values
    df['CTR'] = (((df.click / df.impressions) * 100).round(2)).astype(str) + '%'

    return df.sort_values(["CTR"], ascending=False)

print(CTR("weekday"))


#2. constant bidding strategy
# calculating constant bid stategy

def constant_bidding(bid):
    impression = 0.0
    clicks = 0
    cost = 0.0
    budget = 25000

    for click, pay_price in validation[['click', 'payprice']].values:
        if bid > pay_price:
            impression += 1.0
            clicks += click
            cost += pay_price
        if cost >= budget:
            break
    return impression, clicks, cost


bids = pd.DataFrame()
bids['constants'] = [10, 25, 50, 100, 150, 200, 250, 300]

imp = []
clk = []
costs = []
for i in bids['constants']:
    [imps, clicks, cost] = constant_bidding(i)
    imp.append(imps)
    clk.append(clicks)
    costs.append(cost)
bids['impression_won'] = imp
bids['clicks'] = clk
bids['cost'] = costs
bids['CTR'] = (bids.clicks / bids.impression_won * 100).round(2).astype(str)
bids['CPM'] = (bids.cost / bids.impression_won * 1000).round(2).astype(str)
bids['CPC'] = (bids.cost / bids.clicks).round(2).astype(str)


#print(bids.sort_values("CTR",ascending = False))



#random bidding strategy

from random import randrange


def random_bid(bid):
    impression = 0.0
    clicks = 0
    cost = 0.0
    budget = 25000
    bid = randrange(0, bid)

    for click, pay_price in validation[['click', 'payprice']].values:
        if bid > pay_price:
            impression += 1.0
            clicks += click
            cost += pay_price
        if cost >= budget:
            break
    return impression, clicks, cost, bid


bids = pd.DataFrame()

bids['random'] = [10, 25, 50, 100, 150, 200, 250, 300]

imp = []
clk = []
costs = []
true_random_bid = []
for i in bids['random']:
    [imps, clicks, cost, bid] = random_bid(i)
    imp.append(imps)
    clk.append(clicks)
    costs.append(cost)
    true_random_bid.append(bid)
bids['true_bid'] = true_random_bid
bids['impression_won'] = imp
bids['clicks'] = clk
bids['cost'] = costs
bids['CTR'] = (bids.clicks / bids.impression_won * 100).round(2).astype(str)
bids['CPM'] = (bids.cost / bids.impression_won * 1000).round(2).astype(str)
bids['CPC'] = (bids.cost / bids.clicks).round(2).astype(str)


#print(bids.sort_values("CTR",ascending= False))

#3.linear bidding logistic regression
trainx = train.drop(['click','bidid','logtype','userid','IP','domain',
                'url','urlid','slotid','creative','bidprice','payprice','keypage','city','region'], axis=1)
trainy = train.click

valx = validation.drop(['click','bidid','logtype','userid','IP','domain',
                'url','urlid','slotid','creative','bidprice','payprice','keypage','city','region'], axis=1)
valy = validation.click
testx = test.drop(['bidid','logtype','userid','IP','domain',
                'url','urlid','slotid','creative','keypage','city','region'], axis=1)


def split_useragent(data):
    OS = []
    browser = []
    for i in data.useragent:
        OS.append(i.split("_")[0])
        browser.append(i.split("_")[1])
    data["OS"] = OS
    data["browser"] = browser
    data = data.drop("useragent", axis=1)
    return data

def slot_price(column):
    slotprice = int(column)
    if slotprice > 100:
        return '101+'
    elif slotprice >50:
        return '100-51'
    elif slotprice >10:
        return '50 - 10'
    elif slotprice >0:
        return '10-1'
    else:
        return "0"


def one_hot_encode(variable, data):
    data = pd.concat([data, pd.get_dummies(data[variable], prefix=variable)], axis=1)
    data = data.drop(variable, axis=1)
    return data


def encode(data):
    data = one_hot_encode("hour",data)
    data = one_hot_encode("weekday",data)
    data = one_hot_encode("OS",data)
    data = one_hot_encode("browser",data)
    data = one_hot_encode("adexchange",data)
    data = one_hot_encode("slotwidth",data)
    data = one_hot_encode("slotheight",data)
    data = one_hot_encode("slotvisibility",data)
    data = one_hot_encode("slotformat",data)
    data = one_hot_encode("slotprice",data)
    data = one_hot_encode("advertiser",data)
    return data

def encode_usertag(data):
    temp = pd.DataFrame(data.usertag.str.split(',').tolist())
    df = pd.DataFrame(temp)
    temp_df = pd.get_dummies(df,prefix='usertag')
    tag_df = temp_df.groupby(temp_df.columns, axis=1).sum()
    data = pd.concat([data, tag_df], axis=1)
    data = data.drop('usertag', axis=1)
    return data

trainx = split_useragent(trainx)
valx =split_useragent(valx)

trainx["slotprice"]= trainx["slotprice"].apply(lambda x: slot_price(x))
valx["slotprice"]= valx["slotprice"].apply(lambda x: slot_price(x))

# encode all columns
trainx = encode(trainx)
valx = encode(valx)

#encode usertags
trainx = encode_usertag(trainx)
valx = encode_usertag(valx)

model = linear_model.LogisticRegression(class_weight = "balanced",penalty = "l2")
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(model, param_grid)
clf.fit(trainx,trainy)
print(clf.best_score_)
print(clf.best_estimator_)

#fit model with best c on validation set
# model = linear_model.LogisticRegression(class_weight = "balanced",penalty = "l2",C = 0.001)
# pred = model.fit(trainx,trainy)
# probability = pred.predict_proba(valx)
# pClick = pd.DataFrame(probability)
# pred_label = pred.predict(valx)
# label = pd.DataFrame(pred_label)
#
# print(pClick)
# print(label)