#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[2]:


df = pd.read_csv('../input/weather-prediction/seattle-weather.csv')
df


# In[3]:


df.isnull().sum()


# In[7]:


fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x='date',y='precipitation',data=df,ax=ax)


# In[11]:


fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x='date',y='wind',data=df,ax=ax)


# In[13]:


fig, ax = plt.subplots(figsize=(20, 8))
sns.scatterplot(x='temp_max',y='precipitation',data=df,ax=ax)


# In[14]:


fig, ax = plt.subplots(figsize=(20, 8))
sns.scatterplot(x='temp_min',y='precipitation',data=df,ax=ax)


# In[19]:


fig, ax = plt.subplots(figsize=(20, 8))
sns.barplot(x='weather',y='precipitation',data=df,ax=ax)


# In[20]:


fig, ax = plt.subplots(figsize=(20, 8))
sns.barplot(x='weather',y='temp_max',data=df,ax=ax)


# In[18]:


fig, ax = plt.subplots(figsize=(20, 8))
sns.barplot(x='weather',y='temp_min',data=df,ax=ax)


# **Labelling weather**

# In[21]:


l = LabelEncoder()


# In[22]:


df['weather']=l.fit_transform(df['weather'])


# In[98]:


df['weather']


# In[24]:


df['weather'].value_counts()


# # **Train Test Split and Scaling**

# In[25]:


x=df.drop(['date','weather'],axis=1)
y=df['weather']


# In[26]:


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=42)


# In[28]:


sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)


# # **Training & Prediction**

# **Random Forest and Cross val score**

# In[53]:


rfc = RandomForestClassifier(n_estimators=200)


# In[60]:


rfc.fit(xtrain,ytrain)


# In[61]:


rfc.score(xtest,ytest)


# In[62]:


rfc2 = cross_val_score(estimator = rfc, X = xtrain, y = ytrain, cv = 10)
rfc2.mean()


# In[96]:


yp = rfc.predict(xtest)
c=confusion_matrix(ytest,yp)
fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(c,ax=ax)


# In[97]:


c


# **SVM and GridSearchCV**

# In[33]:


sv = SVC()


# In[34]:


sv.fit(xtrain,ytrain)


# In[35]:


sv.score(xtest,ytest)


# In[83]:


model = GridSearchCV(sv,{
    'C':[0.1,0.4,0.8,1.0,1.2,1.5,2.0,3.0,5.0,8.0,9.0],
    'gamma':[0.1,0.4,0.8,1.0,1.2,1.5,2.0,3.0],
    'kernel':['rbf','linear']
},scoring='accuracy', cv=10)


# In[84]:


model.fit(xtrain,ytrain)
model.best_params_


# In[87]:


model2 = SVC(C=8,kernel='linear',gamma=0.1)
model2.fit(xtrain,ytrain)


# In[88]:


model2.score(xtest,ytest)


# In[92]:


yp = model2.predict(xtest)
c=confusion_matrix(ytest,yp)
fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(c,ax=ax)


# In[93]:


c


# **Logistic Regression**

# In[41]:


lr = LogisticRegression()


# In[42]:


lr.fit(xtrain,ytrain)


# In[43]:


lr.score(xtest,ytest)


# In[94]:


yp = lr.predict(xtest)
c=confusion_matrix(ytest,yp)
fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(c,ax=ax)


# In[95]:


c


# # Best Performer is Random Forest

# # **Predicting sun rain and snow only**

# In[133]:


index=df[df['weather']==0].index
df1 = df.drop(index=index,axis=0)
df1['weather']


# In[134]:


index=df1[df1['weather']==1].index
df1 = df1.drop(index=index,axis=0)
df1['weather']


# In[135]:


x=df1.drop(['date','weather'],axis=1)
y=df1['weather']
y.value_counts()


# In[136]:


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=42)


# In[138]:


sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)


# In[139]:


rfc = RandomForestClassifier(n_estimators=200)


# In[140]:


rfc.fit(xtrain,ytrain)


# In[141]:


rfc.score(xtest,ytest)


# In[142]:


rfc2 = cross_val_score(estimator = rfc, X = xtrain, y = ytrain, cv = 10)
rfc2.mean()


# **Almost 95% Accuracy if fog and drizzle are eliminated which caused variations**

# In[143]:


yp = rfc.predict(xtest)
c=confusion_matrix(ytest,yp)
fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(c,ax=ax)


# In[144]:


c

