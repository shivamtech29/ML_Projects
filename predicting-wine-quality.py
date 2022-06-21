#!/usr/bin/env python
# coding: utf-8

# In[123]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[166]:


df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head(3)


# # Analysis

# In[125]:


df


# In[126]:


df['quality'].value_counts()


# In[127]:


df.columns[df.isna().any()]


# No NA values

# Using SVM

# In[167]:


for i,row in df.iterrows():
    val = row['quality']
    if val <= 6:
        df.at[i,'quality']=0
    else:
        df.at[i,'quality']=1


# In[168]:


from sklearn.model_selection import train_test_split


# In[169]:


x=df.drop(['quality'],axis=1)
y=df['quality']
y.value_counts()


# In[170]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)


# In[171]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)


# In[133]:


from sklearn.svm import SVC


# In[134]:


reg=SVC()


# In[135]:


reg.fit(xtrain,ytrain)


# In[136]:


yp = reg.predict(xtest)


# In[137]:


from sklearn.metrics import confusion_matrix


# In[138]:


c=confusion_matrix(ytest,yp)


# In[139]:


sns.heatmap(c)


# In[140]:


from sklearn.metrics import classification_report
print(classification_report(ytest, yp))


# In[141]:


reg.score(xtest,ytest)


# # GRID SEARCH CV

# In[177]:


from sklearn.model_selection import GridSearchCV


# In[149]:


model = GridSearchCV(reg,{
    'C':[0.1,0.4,0.8,1.0,1.2,1.5],
    'gamma':[0.1,0.4,0.8,1.0,1.2,1.5],
    'kernel':['rbf','linear']
},scoring='accuracy', cv=10)


# In[150]:


model.fit(xtrain,ytrain)


# In[151]:


model.best_params_


# In[160]:


mod2 = SVC(C=1.2,gamma=0.9,kernel='rbf')


# In[161]:


mod2.fit(xtrain,ytrain)


# In[163]:


yp=mod2.predict(xtest)
c=confusion_matrix(ytest,yp)


# In[164]:


sns.heatmap(c)


# In[162]:


mod2.score(xtest,ytest)


# In[165]:


print(classification_report(ytest, yp))


# # RANDOM FOREST

# In[172]:


from sklearn.ensemble import RandomForestClassifier


# In[180]:


rfm = RandomForestClassifier(n_estimators=200)


# In[181]:


rfm.fit(xtrain,ytrain)


# In[182]:


rfm.score(xtest,ytest)


# In[176]:


yp=rfm.predict(xtest)
c=confusion_matrix(ytest,yp)
c


# In[183]:


from sklearn.model_selection import cross_val_score


# In[186]:


rfm2 = cross_val_score(estimator = rfm, X = xtrain, y = ytrain, cv = 10)


# In[187]:


rfm2.mean()


# In[ ]:




