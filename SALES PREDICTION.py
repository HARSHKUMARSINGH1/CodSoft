#!/usr/bin/env python
# coding: utf-8

# # SALES PREDICTION

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('advertising.csv')


X = df.drop('Sales', axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

new_data = {
    'TV': [150, 200],
    'Radio': [25, 30],
    'Newspaper': [50, 60]
}

new_df = pd.DataFrame(new_data)
new_predictions = model.predict(new_df)
print("Predicted Sales for New Data:")
for i, prediction in enumerate(new_predictions):
    print(f"Sample {i+1}: {prediction:.2f}")


# In[2]:


# Scatter plot of actual vs. predicted sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.show()


# In[3]:


# Regression plot
sns.regplot(x=y_test, y=y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Regression Plot: Actual vs. Predicted Sales')
plt.show()


# In[ ]:




