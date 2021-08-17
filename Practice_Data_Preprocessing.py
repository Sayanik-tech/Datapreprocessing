#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[20]:


dataset=pd.read_csv('Data.csv')
print(dataset)


# In[21]:


dataset.info()


# In[22]:


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(x)


# In[23]:


print(y)


# In[24]:


# TAKING CARE OF MISSING VALUES


# In[25]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
print(x)


# In[26]:


# Encoding Independent Variable


# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])],remainder = 'passthrough')
x=np.array(ct.fit_transform(x))
print(x)


# In[28]:


# Encoding Dependent Variable


# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)


# In[30]:


# Splitting the Data


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)


# In[32]:


print(x_test)


# In[33]:


print(x_train)


# In[34]:


print(y_test)


# In[35]:


print(y_train)


# In[36]:


# Feature Selection


# In[39]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])


# In[40]:


print(x_train)


# In[42]:


print(x_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




