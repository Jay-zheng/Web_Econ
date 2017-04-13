import numpy as np
import pandas as pd
import itertools

from subprocess import check_output
print(check_output(["ls", "/Users/Hermione/MasterUCL/Web economics/Assignment/dataset"]).decode("utf8"))

#Load the data
train = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/train.csv")
test = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/test.csv")
validation = pd.read_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/validation.csv")

columns_to_train = ["weekday", "hour", "useragent","region","city",
                    "adexchange","slotwidth","slotheight","slotvisibility","slotformat","slotprice",
                    "advertiser","usertag"]



def slot_price(column):
    slotprice = int(column)
    if slotprice >= 300:
        return '300+'
    elif slotprice >=200:
        return '200+'
    elif slotprice >=100:
        return '100+'
    elif slotprice >=50:
        return '50+'
    else:
        return "50-"

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



def encode_onehot(df,cols):
    enc_data = pd.get_dummies(df[cols],prefix=cols)
    df = df.drop(cols, axis = 1)
    df = df.join(enc_data)
    return df



def onehotencode(data):
    data = encode_onehot(data,"weekday")
    data = encode_onehot(data,"hour")
    data = encode_onehot(data,"region")
    data = encode_onehot(data,"city")
    data = encode_onehot(data,"adexchange")
    data = encode_onehot(data,"slotvisibility")
    data = encode_onehot(data,"slotformat")
    data = encode_onehot(data,"slotprice")
    data = encode_onehot(data,"advertiser")
    data = encode_onehot(data,"slot")
    data = encode_onehot(data,"OS")
    data = encode_onehot(data,"browser")
    return data




#onehotencode usertag
def get_usertaglist(data):
    usertag = data.usertag.str.split(",").tolist()
    usertag = list(itertools.chain.from_iterable(usertag))
    usertag = set(usertag)
    usertaglist = list(usertag)
    return usertaglist



def encodeusertag(data):
    usertaglist = get_usertaglist(data)

    usertagdf = data["usertag"]
    usertagdf = usertagdf.str.split(",")

    a = np.zeros(shape=(usertagdf.count(), len(usertaglist)))
    emptyusertag = pd.DataFrame(a, columns=usertaglist)


    for ii in usertagdf.index:
        component = usertagdf.loc[ii]
        for tag in component:
            cols = tag
            ind = usertaglist.index(cols)
            encoded = emptyusertag
            encoded.ix[ii,ind]=1.0
    return encoded



def getcleandata(data):
    data1 = preprocessing(data)
    data2 = onehotencode(data1)
    usertag = encodeusertag(data2)
    cleandata = data2.drop("usertag", axis = 1).join(usertag)
    return cleandata

traindata = pd.DataFrame(train[columns_to_train])
testdata = pd.DataFrame(test[columns_to_train])
validationdata = pd.DataFrame(validation[columns_to_train])


cleantrain = getcleandata(traindata)
cleantest = getcleandata(testdata)
cleanval = getcleandata(validationdata)

cleantrain.to_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/cleantrain.csv")
cleantest.to_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/cleantest.csv")
cleanval.to_csv("/Users/Hermione/MasterUCL/Web economics/Assignment/dataset/cleanvalidation.csv")

