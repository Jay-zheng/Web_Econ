
# coding: utf-8

# In[4]:

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


# In[5]:

#loading data
train = pd.read_csv("Desktop/Webeco/dataset/train.csv")
test = pd.read_csv("Desktop/Webeco/dataset/test.csv")
validation = pd.read_csv("Desktop/Webeco/dataset/validation.csv")


# In[6]:

train.columns


# In[7]:

#keep features that only need to be considered when training 
columns_to_train = ["weekday", "hour", "useragent","city","region",
                    "adexchange","slotwidth","slotheight","slotvisibility","slotformat","slotprice",
                    "advertiser","usertag"]


# In[8]:

#seperate slot prices into various buckets
def slot_price(column):
    slotprice = int(column)
    if slotprice >= 200:
        return '200+'
    elif slotprice >=100:
        return '199 - 100'
    elif slotprice >=50:
        return '99 - 50'
    elif slotprice >=10:
        return '49 - 10'
    else:
        return "10-"


def slot_size(data):
    data["slot"]=data["slotwidth"].astype(str) + "*" + data["slotheight"].astype(str)
    data = data.drop(["slotheight","slotwidth"],1)
    return data

def split_useragent(data):
    data["OS"] = data["useragent"].apply(lambda x: x.split("_")[0])
    data["browser"]=data["useragent"].apply(lambda x: x.split("_")[1])
    data = data.drop("useragent",1)
    return data

def preprocessing(df):
    df["slotprice"] = df["slotprice"].apply(lambda x: slot_price(x))
    df = slot_size(df)
    df = split_useragent(df)
    return df


def onehot_encode(df,columns):
    enc_data = pd.get_dummies(df[columns],prefix=columns)
    df = df.drop(columns, axis = 1)
    df = df.join(enc_data)
    return df

def onehotencode(data):
    data = onehot_encode(data,"weekday")
    data = onehot_encode(data,"hour")
    data = onehot_encode(data,"city")
    data = onehot_encode(data,"region")
    data = onehot_encode(data,"adexchange")
    data = onehot_encode(data,"slotvisibility")
    data = onehot_encode(data,"slotformat")
    data = onehot_encode(data,"slotprice")
    data = onehot_encode(data,"advertiser")
    data = onehot_encode(data,"slot")
    data = onehot_encode(data,"OS")
    data = onehot_encode(data,"browser")
    return data




#onehotencode usertag
def get_usertaglist(data):
    usertag = data.usertag.str.split(",").tolist()
    usertag = list(itertools.chain.from_iterable(usertag))
    usertag = set(usertag)
    usertaglist = list(usertag)
    return usertaglist

def encode_usertag(data):
    temp = pd.DataFrame(data.usertag.str.split(',').tolist())
    df = pd.DataFrame(temp)
    temp_df = pd.get_dummies(df,prefix='usertag')
    tag_df = temp_df.groupby(temp_df.columns, axis=1).sum()
    data = pd.concat([data, tag_df], axis=1)
    data = data.drop('usertag', axis=1)
    return data


def processingdata(data):
    data1 = preprocessing(data)
    data2 = onehotencode(data1)
    usertag = encode_usertag(data2)
    cleandata = data2.drop("usertag", axis = 1).join(usertag)
    return cleandata


# In[ ]:

traindata = pd.DataFrame(train[columns_to_train])
testdata = pd.DataFrame(test[columns_to_train])
validationdata = pd.DataFrame(validation[columns_to_train])


# In[ ]:

encodedtrain = processingdata(traindata)
encodedtest = processingdata(testdata)
encodedvalidation = processingdata(validationdata)


# In[ ]:

encodedtrain.to_csv("Desktop/data/cleantrain.csv")
encodedtest.to_csv("Desktop/data/cleantest.csv")
encodedvalidation.to_csv("Desktop/data/cleanvalidation.csv")


# In[ ]:



