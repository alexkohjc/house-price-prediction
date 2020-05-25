#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


train_df = pd.read_csv('input/train.csv')
#test_df = pd.read_csv('input/test.csv')


# In[6]:


cols = ['OverallQual', 'GrLivArea', 'GarageCars',
       'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
       'TotRmsAbvGrd', 'YearBuilt', 'KitchenAbvGr', 'EnclosedPorch',
       'MSSubClass', 'OverallCond', 'YrSold', 'LowQualFinSF', 'Id',
       'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', 'SalePrice']


# In[7]:


train_df = train_df[cols]


# In[9]:


X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Model Fitting

# In[37]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# (Multiple) Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Support Vector Regression
supp_vec_reg = SVR(kernel = 'rbf')
supp_vec_reg.fit(X_train, y_train)

# Decision Trees
d_tree = DecisionTreeRegressor(random_state = 0)
d_tree.fit(X_train, y_train)

# Random Forest
r_forest = RandomForestRegressor(n_estimators = 10, random_state = 0)
r_forest.fit(X_train, y_train)


# # Calculating RMSE

# In[51]:


from sklearn.metrics import mean_squared_error

y_pred_lin_reg = lin_reg.predict(X_test)
y_pred_supp_vec_reg = supp_vec_reg.predict(X_test)
y_pred_d_tree = d_tree.predict(X_test)
y_pred_r_forest = r_forest.predict(X_test)

print("Linear Reg: " + "{:.2f}".format(mean_squared_error(y_test, y_pred_lin_reg, squared=False)))
print("Support Vector Reg: " + "{:.2f}".format(mean_squared_error(y_test, y_pred_supp_vec_reg, squared=False)))
print("Decision Tree: " + "{:.2f}".format(mean_squared_error(y_test, y_pred_d_tree, squared=False)))
print("Random Forest: " + "{:.2f}".format(mean_squared_error(y_test, y_pred_r_forest, squared=False)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




