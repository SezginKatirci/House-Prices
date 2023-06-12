
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:54:32 2023

@author: Sezgin Katırcı
"""

import numpy as np
import pandas as pd

traindata=pd.read_csv("C:\\Users\\Dell\\Desktop\\House price predict\\train.csv")
traindata.drop("Id",axis=1,inplace=True)

tdatacorr=traindata.corr()
trainnull=traindata.isnull().sum()


s=traindata.shape
c=s[1]
r=s[0]
i=0
for x in traindata:
    if traindata[x].dtype == 'object':
        n=traindata[x].isnull()
        j=0
        while j<r:       
            if n[j]==True:
                traindata[x][j]=traindata.columns[i]+"Null"
            j=j+1
    i=i+1

j=0
a=traindata["GarageYrBlt"].isnull()
b=traindata["LotFrontage"].isnull()
c=traindata["LotFrontage"].mean()
while j<len(traindata):       
    if a[j]==True:
        traindata["GarageYrBlt"][j]=traindata["YearBuilt"][j]
    if b[j]==True:
        traindata["LotFrontage"][j]=c    
    j=j+1
traindata.rename(columns={"LwQ":"LwQ1"},inplace=True)

from sklearn import preprocessing

x=0
c=0
ntd=pd.DataFrame()
while x<80:
    td1=traindata.iloc[:,x:x+1].values       
    if td1[:,0].dtype == 'object':  
        settd1=set()
        i=0
        while i<len(td1):
            settd1.add(td1[i][0]+str(x))
            i=i+1          
        ohe=preprocessing.OneHotEncoder()        
        td1=ohe.fit_transform(td1,).toarray()
        td1=pd.DataFrame(data=td1,columns=settd1)
        ntd=pd.concat([ntd,td1],axis=1)
        c=c+len(settd1)
    else:
        td1=pd.DataFrame(td1)        
        ntd=pd.concat([ntd,td1],axis=1)
        ntd.rename(columns={0:traindata.columns[x]},inplace=True)
        c=c+1
    x=x+1    

ntdcorr=ntd.corr()
ntdnull=ntd.isnull().sum()

for x in ntd:
    o=1
    for y in ntd:
        if y == x:
            ntd.rename(columns={x:y+str(o)},inplace=True)
            o=o+1
      

ntdcorr2=ntd.corr()
ntd.drop("BLQ341",axis=1,inplace=True)

print(len(ntd))
ntd.dropna(axis=0,inplace=True)
print(len(ntd))
ntdnull=ntd.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sbn

plt.figure(figsize=(7,5))
sbn.distplot(ntd["SalePrice1"])
plt.show()
sbn.scatterplot(x="GrLivArea1",y="SalePrice1",data=ntd)
plt.show()

ntd=ntd.sort_values("SalePrice1",ascending=False).iloc[15:]
ntd=ntd.sort_values("GrLivArea1",ascending=False).iloc[5:]

plt.figure(figsize=(7,5))
sbn.distplot(ntd["SalePrice1"])
plt.show()
sbn.scatterplot(x="GrLivArea1",y="SalePrice1",data=ntd)
plt.show()

x=ntd.iloc[:,0:303].values
y=ntd.iloc[:,303:].values

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=30,random_state=0) 
rf_reg.fit(x,y.ravel())

from sklearn.metrics import r2_score
print(r2_score(y,rf_reg.predict(x)))

""" Test """

testdata=pd.read_csv("C:\\Users\\Dell\\Desktop\\House price predict\\test.csv")
testdataID=testdata.iloc[:,0:1]
testdata=testdata.iloc[:,1:]
#testdata.drop("Id",axis=1,inplace=True)
s=testdata.shape
c=s[1]
r=s[0]
i=0
for x in testdata:
    if testdata[x].dtype == 'object':
        n=testdata[x].isnull()
        j=0
        while j<r:       
            if n[j]==True:
                testdata[x][j]=testdata.columns[i]+"Null"
            j=j+1
    i=i+1

testdata.rename(columns={"LwQ":"LwQ1"},inplace=True)

from sklearn import preprocessing

x=0
c=0
ntd2=pd.DataFrame()
while x<78:
    td1=testdata.iloc[:,x:x+1].values       
    if td1[:,0].dtype == 'object':  
        settd1=set()
        i=0
        while i<len(td1):
            settd1.add(td1[i][0]+str(x))
            i=i+1          
        ohe=preprocessing.OneHotEncoder()        
        td1=ohe.fit_transform(td1).toarray()
        td1=pd.DataFrame(data=td1,columns=settd1)
        ntd2=pd.concat([ntd2,td1],axis=1)
        c=c+len(settd1)
    else:
        td1=pd.DataFrame(td1)        
        ntd2=pd.concat([ntd2,td1],axis=1)
        ntd2.rename(columns={0:testdata.columns[x]},inplace=True)
        c=c+1
    x=x+1    

ntd2.drop("BLQ34",axis=1,inplace=True)
for x in ntd2:
    o=1
    for y in ntd2:
        if y == x:
            ntd2.rename(columns={x:y+str(o)},inplace=True)
            o=o+1

ntd2.fillna(0,inplace=True)
ntd2null=ntd2.isnull().sum()
ntd2corr=ntd2.corr()
ntdcolumns=list(ntd.columns)
ntd2columns=list(ntd2.columns)
i=0
eklenecek=[]
while i<len(ntdcolumns):
    if ntd2columns.count(ntdcolumns[i])==0:
        eklenecek.append([i,ntdcolumns[i]])
    i=i+1

i=0
cikarilacak=[]
while i<len(ntd2columns):
    if ntdcolumns.count(ntd2columns[i])==0:
        cikarilacak.append([i,ntd2columns[i]])
    i=i+1
eklenecek.remove([303,"SalePrice1"])

for x, y in cikarilacak:
    ntd2.drop(y,axis=1,inplace=True)

ptd=pd.DataFrame()

z=0
while z<len(eklenecek):
    x=eklenecek[z][0]
    y=eklenecek[z][1]      
    if len(ptd)==0:
        ilk=ntd2.iloc[:,:x]
        son=ntd2.iloc[:,x:]
        sifir=pd.DataFrame(np.zeros(len(ntd2)))        
        ptd=pd.concat([ilk,sifir],axis=1)
        ptd.rename(columns={0:y},inplace=True)
        ptd=pd.concat([ptd,son],axis=1)
    else:
        ilk=ptd.iloc[:,:x]
        son=ntd2.iloc[:,x-z:]
        sifir=pd.DataFrame(np.zeros(len(ntd2)))        
        ptd=pd.concat([ilk,sifir],axis=1)
        ptd.rename(columns={0:y},inplace=True)
        ptd=pd.concat([ptd,son],axis=1)
    z=z+1


PredictList=[]
i=0
while i<len(ptd):
    PredictList.append([int(testdataID.iloc[i]),float(rf_reg.predict(ptd.iloc[i:i+1,0:]))])
    i=i+1

PredictList=pd.DataFrame(PredictList)
PredictList.to_csv("C:\\Users\\Dell\\Desktop\\PredictList.csv",header=["Id","SalePrice"],index=False)


















