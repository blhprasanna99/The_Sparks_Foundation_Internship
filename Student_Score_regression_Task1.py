#!/usr/bin/env python
# coding: utf-8

# ## Author : Lakshmi Hari Prasanna Bollineni
# 
# 
# ## Data Science and Business Analytics Internship
# 
# 
# ## GRIP - The Spark Foundation
# 
# 
# ## Task 1 : Predict the percentage of an student based on the no. of study hours.
# 
# 
# ### Description : In this task, we will predict the marks of the student depending upon the number of hours he/she                                studied. This is a linear regression task hat involves two variables.

# ## Importing Libraries and DataSet

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df= pd.read_csv("C:/Users/Hari/Desktop/Hours.csv")


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.shape


# In[10]:


df["Scores"].plot.hist()


# In[11]:


df.plot.scatter(x="Hours",y="Scores",title="Hours and Scores")


# ## Preparing Data

# In[12]:


X=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=0)


# ## Training and Testing the Algorithm

# In[14]:


from sklearn.linear_model import LinearRegression
l_regressor = LinearRegression()


# In[15]:


l_regressor.fit(X_train,y_train)
print("The Training of Linear regression model is completed")


# In[16]:


regression_line= l_regressor.coef_*X+l_regressor.intercept_   # This is line equation mx+c

plt.scatter(X,y)
plt.plot(X,regression_line)
plt.show()


# In[17]:


#testing
y_pred=l_regressor.predict(X_test)


# In[18]:


comparision= pd.DataFrame({'Actual' : y_test,'Predicted': y_pred})
print(comparision)


# ## Model Evaluation

# In[19]:


from sklearn import metrics
import math
print("Mean absolute error :",metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared error : ",metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared error :",math.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# ## User Input

# In[38]:


print("Enter the number of hours student studied")
hr=float(input())
arr= np.array(hr)
hrs=arr.reshape(-1,1)
o_p= l_regressor.predict(hrs)
print("Number of hours studied={}".format(hr))
print("Percentage Score expected = {}".format(o_p[0]))


# In[ ]:




