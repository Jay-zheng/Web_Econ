
# coding: utf-8

# In[1]:

import random
import numpy as np
import pandas as pd


# In[2]:

train = pd.read_csv("Desktop/Webeco/dataset/train.csv")
validation = pd.read_csv("Desktop/Webeco/dataset/validation.csv")


# In[3]:

train.columns


# In[4]:

df = pd.DataFrame()
df['advertiser'] = np.sort(train.advertiser.unique())
df['impressions'] = train.groupby('advertiser').size().values
df['click'] = train.groupby('advertiser').click.aggregate(np.sum).values
df['cost'] = train.groupby('advertiser').payprice.aggregate(np.sum).values
df['CTR'] = (((df.click / df.impressions) * 100).round(2)).astype(str) + '%'
df['CPM'] = (((df.cost / df.impressions) * 1000).round(2)).astype(str)
df['eCPC'] = ((df.cost / df.click).round(2)).astype(str)


# In[5]:

df


# In[6]:

def CTR(variable):
    df = pd.DataFrame()
    df[variable] = np.sort(train[variable].unique())
    df['impressions'] = train.groupby(variable).size().values
    df['click'] = train.groupby(variable).click.aggregate(np.sum).values
    df['CTR'] = (((df.click / df.impressions) * 100).round(2)).astype(str) + '%'
    
    return df.sort_values(["CTR"],ascending = False)


# In[7]:

CTR("weekday")


# In[8]:

CTR("hour")


# In[9]:

CTR("adexchange")


# In[10]:

CTR("useragent").head(10)


# In[11]:

CTR("region").head(10)


# In[12]:

CTR("city").head(10)


# In[ ]:



