#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r'C:\Users\UHASI\Downloads\docs\Diamond\diamonds.csv')
data


# In[7]:


figure = px.scatter(data_frame = data, x="carat",
                    y="price", size="depth",color = 'color', trendline="ols")
figure.show()


# In[24]:


data["size"] = data["x"] * data["y"] * data["z"]
data


# In[11]:


figure = px.scatter(data_frame = data, x="size",
                    y="price", size="size", 
                    color= "cut", trendline="ols")
figure.show()


# In[12]:


fig = px.box(data, x="cut", 
             y="price", 
             color="color")
fig.show()


# In[13]:


fig = px.box(data, 
             x="cut", 
             y="price", 
             color="clarity")
fig.show()


# In[15]:


correlation = data.corr()
correlation["price"].sort_values(ascending=False)


# In[27]:


data["cut"] = data["cut"].map({"Ideal": 1, 
                               "Premium": 2, 
                               "Good": 3,
                               "Very Good": 4,
                               "Fair": 5})


# In[28]:


from sklearn.model_selection import train_test_split
x = np.array(data[["carat", "cut", "size"]])
y = np.array(data[["price"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)


# In[29]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(xtrain, ytrain)


# In[30]:


print("Diamond Price Prediction")
A = float(input("Carat Size: "))
B = int(input("Cut Type (Ideal: 1, Premium: 2, Good: 3, Very Good: 4, Fair: 5): "))
C = float(input("Size: "))
features = np.array([[A, B, C]])
print("Predicted Diamond's Price = ", model.predict(features))


# In[ ]:




