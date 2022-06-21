#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Applying ANN
--> ANN  86.8%
--> Logistic Regression 88.5%
--> SVM  88.5%
--> KNN  90.1%

"""


# In[ ]:


# Importing modules

import numpy as np 
import pandas as pd 

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


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
from sklearn.metrics import accuracy_score


# In[ ]:


df = pd.read_csv('../input/heart-attack-analysis-prediction-dataset/heart.csv')
df


# In[ ]:


dfo = pd.read_csv('../input/heart-attack-analysis-prediction-dataset/o2Saturation.csv')
#dfo


# In[ ]:


df.isnull().sum()


# # **EDA and Plotting**

# In[ ]:


sns.scatterplot(x='age',y='chol',data=df)


# In[ ]:


sns.barplot(x='cp',y='age',data=df)


# In[ ]:


sns.scatterplot(x='trtbps',y='chol',data=df)


# In[ ]:


sns.scatterplot(x='trtbps',y='thalachh',data=df)


# # **Train Test Split and Scaling**

# In[ ]:


x=df.drop(['output'],axis=1)
y=df['output']


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)


# # **DL ANN**

# In[ ]:


clf = Sequential()


# In[ ]:


clf.add(Dense(input_dim=13,units=13,kernel_initializer='he_uniform',activation='relu'))
clf.add(Dropout(0.3))
clf.add(Dense(units=13,kernel_initializer='he_uniform',activation='relu'))
clf.add(Dropout(0.3))
clf.add(Dense(units=13,kernel_initializer='he_uniform',activation='relu'))
clf.add(Dropout(0.3))
clf.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
clf.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


model = clf.fit(xtrain,ytrain,validation_split=0.2,epochs=100,batch_size=10)


# In[ ]:


yp=clf.predict(xtest)
yp=yp>0.5


# In[ ]:


score=accuracy_score(yp,ytest)
score


# In[ ]:


print(model.history.keys())


# In[ ]:





# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # **Logistic Regression , SVM , KNN**

# In[ ]:


lr=LogisticRegression()
lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)


# In[ ]:


sv=SVC(C=10,gamma=0.2,kernel='linear')
sv.fit(xtrain,ytrain)
sv.score(xtest,ytest)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(xtrain,ytrain)
kn.score(xtest,ytest)


# In[ ]:





# In[ ]:





# In[ ]:




