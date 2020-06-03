

# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random
import seaborn as sns
from fbprophet import Prophet

# dataframes creation for both training and testing datasets 
df=pd.read_csv('avocado.csv')
# 
# - Date: The date of the observation
# - AveragePrice: the average price of a single avocado
# - type: conventional or organic
# - year: the year
# - Region: the city or region of the observation
# - Total Volume: Total number of avocados sold
# - 4046: Total number of avocados with PLU 4046 sold
# - 4225: Total number of avocados with PLU 4225 sold
# - 4770: Total number of avocados with PLU 4770 sold

# Let's view the head of the training dataset
df.head()
# Let's view the last elements in the training dataset
df.tail(10)
df.describe()
df.info()

df.isnull().sum()


# # TASK #3: EXPLORE DATASET  

df=df.sort_values('Date')

# Plot date and average price
plt.figure(figsize=(10,10))
plt.plot(df['Date'],df['AveragePrice'])

# Plot distribution of the average price
plt.figure(figsize=(10,6))
sns.distplot(df['AveragePrice'])

# Plot a violin plot of the average price vs. avocado type
sns.violinplot(y='AveragePrice',x='type',data=df)

# Bar Chart to indicate the number of regions 

sns.set(font_scale=0.7) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'region', data = df)
plt.xticks(rotation = 45)

# Bar Chart to indicate the count in every year
sns.set(font_scale=1.5) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'year', data = df)
plt.xticks(rotation = 45)

# plot the avocado prices vs. regions for conventional avocados
conventional=sns.catplot('AveragePrice','region',data=df[df['type']=='conventional'],hue='year',height=20)

# plot the avocado prices vs. regions for organic avocados
organic=sns.catplot('AveragePrice','region',data=df[df['type']=='organic'],hue='year',height=20)


# # TASK 4: PREPARE THE DATA BEFORE APPLYING FACEBOOK PROPHET TOOL 
df_sample=df[['Date','AveragePrice']]
df_sample
df_sample=df_sample.rename(columns={'Date':'ds','AveragePrice':'y'})
df_sample


# # TASK 5: DEVELOP MODEL AND MAKE PREDICTIONS - PART A

m=Prophet()
m.fit(df_sample)

# Forcasting into the future
future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)
forecast

figure=m.plot(forecast,xlabel='Date',ylabel='Price')
figure2=m.plot_components(forecast)


# # TASK 6: DEVELOP MODEL AND MAKE PREDICTIONS (REGION SPECIFIC) - PART B

# Select specific region
df_r1=df[df['region']=='West']
df_r2=df[df['region']=='Chicago']

df_r1=df_r1.sort_values('Date')
df_r2=df_r2.sort_values('Date')

plt.plot(df_r1['Date'],df_r1['AveragePrice'])
plt.plot(df_r2['Date'],df_r2['AveragePrice'])

df_r1=df_r1.rename(columns={'Date':'ds','AveragePrice':'y'})
df_r2=df_r2.rename(columns={'Date':'ds','AveragePrice':'y'})

m2= Prophet()
m2.fit(df_r1)
m3= Prophet()
m3.fit(df_r2)

# Forcasting into the future
future2= m2.make_future_dataframe(periods=365)
forecast2 = m2.predict(future)
future3= m3.make_future_dataframe(periods=365)
forecast3 = m3.predict(future)

figure3= m2.plot(forecast2, xlabel='Date', ylabel='Price')
figure4 = m2.plot_components(forecast2)

figure5= m3.plot(forecast3, xlabel='Date', ylabel='Price')
figure6 = m3.plot_components(forecast3)

