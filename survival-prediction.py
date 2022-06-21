#!/usr/bin/env python
# coding: utf-8

# In[129]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
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


# In[130]:


df = pd.read_csv('../input/titanic/train.csv')
testdf = pd.read_csv('../input/titanic/test.csv')
preddf = pd.read_csv('../input/titanic/gender_submission.csv')


# In[131]:


df.head(1)


# In[132]:


df.columns[df.isna().any()]


# In[133]:


testdf.isnull().sum()


# In[134]:


df.isnull().sum()


# In[135]:


df['Embarked']=df['Embarked'].fillna('Q')


# In[136]:


df['Age']=df['Age'].fillna(df.Age.mean())


# In[137]:


sns.barplot(x='Sex',y='Survived',data=df)


# In[138]:


x=testdf['Sex']
y=preddf['Survived']
sns.barplot(x=x,y=y)


# **Test data is completely biased towards female survival**

# In[139]:


sns.lineplot(x='Age',y='Survived',data=df)


# In[140]:


x=testdf['Age']
y=preddf['Survived']
sns.lineplot(x=x,y=y)


# **middle aged people survived less**

# In[141]:


sns.lineplot(x='Fare',y='Survived',data=df)


# In[142]:


x=testdf['Fare']
y=preddf['Survived']
sns.lineplot(x=x,y=y)


# High payers mostly survives

# In[143]:


l=LabelEncoder()
df['Sex']=l.fit_transform(df['Sex'])
testdf['Sex']=l.fit_transform(testdf['Sex'])


# # **Train set and Test set**

# In[144]:


xtrain = df.drop(['Cabin','Survived','Name','Ticket','PassengerId'],axis=1)
ytrain = df['Survived']


# In[145]:


xtest = testdf.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
ytest = preddf['Survived']


# In[146]:


xtrain


# In[147]:


l1=LabelEncoder()
xtrain['Embarked']=l1.fit_transform(xtrain['Embarked'])
xtest['Embarked']=l1.fit_transform(xtest['Embarked'])


# In[148]:


xtest['Age']=xtest['Age'].fillna(xtest.Age.mean())


# In[149]:


xtest.isnull().sum()


# In[150]:


xtest['Fare']=xtest['Fare'].fillna(xtest.Fare.mean())


# # **Training**

# **Random Forest**

# In[179]:


rfc = RandomForestClassifier(n_estimators=200)


# In[185]:


rfc.fit(xtrain,ytrain)


# In[186]:


rfc.score(xtest,ytest)


# In[187]:


rfm2= cross_val_score(estimator=rfc,X=xtrain,y=ytrain,cv=10)
rfm2.mean()


# In[189]:


pred1 = rfc.predict(xtest)


# # **SVM + GridSearchCV**

# In[155]:


sv = SVC()


# In[156]:


sv.fit(xtrain,ytrain)


# In[157]:


sv.score(xtest,ytest)


# In[158]:


gsv = GridSearchCV(sv,{
    'C':[0.1,0.4,0.8,1.0,1.2,1.5,2.0,3.0,5.0,8.0,9.0],
    'gamma':[0.1,0.4,0.8,1.0,1.2,1.5],
    'kernel':['rbf','linear']
},scoring='accuracy', cv=10)


# In[159]:


"""gsv.fit(xtrain,ytrain)"""


# In[160]:


"""gsv.best_params_"""


# In[161]:


"""sv2 = SVC(C=,gamma=,kernel='')"""


# In[162]:


"""sv2.fit(xtrain,ytrain)"""


# In[163]:


"""sv2.score(xtest,ytest)"""


# # **Logistic Regression**

# In[164]:


lr = LogisticRegression()


# In[200]:


xtrain=pd.concat([xtrain,xtest])
ytrain=pd.concat([ytrain,ytest])


# In[201]:


lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)


# In[166]:


"""lr2=cross_val_score(lr,xtrain,ytrain,cv=1, scoring='roc_auc')
lr2.mean()"""


# In[202]:


predictions = lr.predict(xtest)


# In[203]:


submission=testdf[['PassengerId']]
submission


# In[204]:


submission['Survived']=predictions
submission


# In[ ]:





# In[205]:


submission.to_csv("submission5.csv",index=None)


# In[195]:


dfs=pd.read_csv('./submission4.csv')
dfs


# In[190]:


submissionr=testdf[['PassengerId']]
submissionr


# In[191]:


submissionr['Survived']=pred1
submissionr


# In[192]:


submissionr.to_csv("submission3.csv",index=None)


# # **94% accuracy by Logistic Regression**
