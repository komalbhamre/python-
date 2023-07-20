#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score


# In[2]:


df = pd.read_csv('creditcard.csv.zip')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scaler = StandardScaler()


# In[8]:


scaler.fit(df.drop('Class',axis=1))


# In[9]:


x = scaler.transform(df.drop('Class',axis=1))


# In[10]:


x


# In[11]:


y = df.Class


# In[12]:


y


# In[25]:


# Step 3: Splitting the Data
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=40)


# In[26]:


# Step 4: Model Training
# Train the logistic regression model
model = LogisticRegression()


# In[27]:


model.fit(X_train, y_train)


# In[28]:


y_pred = model.predict(X_test)


# In[29]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[30]:


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

