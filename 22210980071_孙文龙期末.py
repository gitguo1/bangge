#!/usr/bin/env python
# coding: utf-8

# ## 第五题

# In[7]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[8]:


df=pd.read_excel('/Users/ziman/Desktop/第五题-购买意愿.xlsx')
df.index=df['User id']
df.drop('User id',axis=1,inplace=True)
df=df.replace({'Age':{'<=30':0, '[31,40]':1, '>40':2 } } )
df=df.replace({'Incoming':{'low':0, 'medium':1, 'high':2}})
df=df.replace({'Student':{'no':0, 'yes':1}})
df=df.replace({'Credit Rating':{'fair':0, 'excellent':1}})
df=df.replace({'Buying':{'no':0 ,'yes':1 }})
X=df.drop('Buying',axis=1)
y=df['Buying']


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='gini',max_depth=3)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

import sklearn.metrics
print(sklearn.metrics.classification_report(y_test,y_pred))


# In[10]:


from sklearn.tree import plot_tree
plt.figure(figsize=(8,6),dpi=100)
print(plot_tree(model,feature_names=X.columns,filled=True))


# In[11]:


need_to_be_predicted=pd.DataFrame(
    {'Age':[2],'Incoming':[1],'Student':[0],'Credit Rating':[1]})
print(model.predict(need_to_be_predicted))


# ## 第九题

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# In[13]:


ham_list=[]
spam_list=[]
for i in range(1,6):
    ham_path='/Users/ziman/Desktop/enron{}/ham/'.format(i)
    ham_dirs = os.listdir(ham_path)
    spam_path='/Users/ziman/Desktop/enron{}/spam/'.format(i)
    spam_dirs = os.listdir(spam_path)
    for file in ham_dirs:
        ham_list.append(ham_path+file)
    ham_list.sort()
    for file in spam_dirs:
        spam_list.append(spam_path+file)
    spam_list.sort()

contents=[]
y=[]
for i in range(len(ham_list)):
    text=open(ham_list[i],errors='ignore').read()
    contents.append(text)
    y.append(0)
for j in range(len(spam_list)):
    text=open(spam_list[j],errors='ignore').read()
    contents.append(text)
    y.append(1)
    
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(contents,y,test_size=0.2,random_state=123)


# In[19]:


pd.DataFrame(contents)


# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer as TV
tfi=TV().fit(X_train)
X_train=tfi.transform(X_train)
X_test=tfi.transform(X_test)


# In[22]:


feature=pd.DataFrame(X_train.toarray(),columns=tfi.get_feature_names())
feature


# In[26]:


pd.DataFrame(X_test.toarray(),columns=tfi.get_feature_names())


# In[27]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
import sklearn.metrics
print(sklearn.metrics.classification_report(y_test,y_pred))


# In[28]:


ham_path='/Users/ziman/Desktop/enron6/ham/'
ham_dirs = os.listdir(ham_path)
ham_list=[]
spam_path='/Users/ziman/Desktop/enron6/spam/'
spam_dirs = os.listdir(spam_path)
spam_list=[]
for file in ham_dirs:
    ham_list.append(ham_path+file)
ham_list.sort()
for file in spam_dirs:
    spam_list.append(spam_path+file)
spam_list.sort()

test_ham=[]
y_ham=[]
test_spam=[]
y_spam=[]
test=[]
y=[]
for i in range(len(ham_list)):
    text=open(ham_list[i],errors='ignore').read()
    test_ham.append(text)
    y_ham.append(0)
    test.append(text)
    y.append(0)
for j in range(len(spam_list)):
    text=open(spam_list[j],errors='ignore').read()
    test_spam.append(text)
    y_spam.append(1)
    test.append(text)
    y.append(1)

test_ham=tfi.transform(test_ham)
test_spam=tfi.transform(test_spam)
test=tfi.transform(test)


# In[29]:


print('enron6总体准确率',model.score(test,y))
print('enron6非垃圾邮件准确率',model.score(test_ham,y_ham))
print('enron6垃圾邮件准确率',model.score(test_spam,y_spam))

