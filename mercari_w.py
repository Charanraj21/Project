# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 19:51:36 2021

@author: Charanraj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.tsv', sep='\t')
train, test = train_test_split(data, test_size=0.2, shuffle=False)
print(train.shape, test.shape)
print('\n')

# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n""There are " + str(mis_val_table_ren_columns.shape[0]) +" columns that have missing values.")
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

print(missing_values_table(data))
print('\n')

print(train.info())
print('\n')

# **Exploratory Data Analysis**
print('There are a total of',train.shape[0],'observations in the train data.')
print('\n')

#this command displays first few rows of the data set
print(train.head())
print('\n')

#Price
print(train['price'].describe())
print('\n')

price = train['price'].values
price = np.sort(price, axis=None)
print('{}th percentile value is {}'.format(90, price[int(len(price)*(float(90)/100))]))
print('\n')

#Checking Distribution of Price variable
train['price'].plot.hist(bins=50, figsize=(10,5), edgecolor='white',range=[0,500])
plt.xlabel('Price', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.tick_params(labelsize=15)
plt.title('Price Distribution - Training Set', fontsize=17)
plt.show()
print('\n')

#Shipping
print(train['shipping'].value_counts(normalize=True)*100)
print('\n')

shipping_fee_by_seller = train.loc[train.shipping==1, 'price']
shipping_fee_by_buyer = train.loc[train.shipping==0, 'price']
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(shipping_fee_by_seller, color='Orange', alpha=1.0, bins=50, range=[0,100], label='Price when Seller pays Shipping')
ax.hist(shipping_fee_by_buyer, color='Green', alpha=1.0, bins=50, range=[0,100], label='Price when Buyer pays Shipping')
ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
plt.xlabel('Price', fontsize=17)
plt.ylabel('Frequency', fontsize=17)
plt.title('Price Distribution by Shipping Type', fontsize=17)
plt.tick_params(labelsize=15)
plt.legend()
plt.show()
print('\n')

print('The median price is ${}'.format(round(shipping_fee_by_seller.median(), 2)), 'if seller pays shipping')
print('The median price is ${}'.format(round(shipping_fee_by_buyer.median(), 2)), 'if buyer pays shipping')
print('\n')


#Item-Condition
print(train['item_condition_id'].value_counts(normalize=True, sort=False)*100)
print('\n')

#for easier visualization, considering the prices from range of 0-100
price_100 = train[train['price']<100]
fig, ax = plt.subplots(figsize=(20,7.5))
sns.boxplot(x='item_condition_id', y='price', data=price_100, ax=ax)
plt.xlabel('Item Condition', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print('\n')


#Conditiom-ID
con = train[(train['item_condition_id']==5) & (train['price']>=20)]
print(con.head(6))
print('\n')


#Category-Name
print((train['category_name'].value_counts(normalize=True)*100).head(6))
print('\n')
print("There are %d unique main categories." % train['category_name'].nunique())

