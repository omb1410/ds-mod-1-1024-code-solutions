# -*- coding: utf-8 -*-
"""visualization_project_laptop_sales.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gNOQc0lSprxSHEe-gBIYAU21tKkgZ_wI
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Sales_data.txt')

df

#1
df.groupby('Contact Sex')['Profit'].sum().plot(kind='bar');
#The target market is Male

#2
df_gender = df.groupby('Contact Sex')
cost_for_gender = df_gender['Profit'].sum()/(df_gender['Our Cost'].sum()+df_gender['Shipping Cost'].sum())
cost_for_gender.plot(kind='bar')
#Male should be targeted if business is cash constrained

#3
cost_for_cust = df_gender['Profit'].sum()/(df_gender['Sale Price'].sum()+df_gender['Shipping Cost'].sum())
cost_for_cust.plot(kind='bar')
#If consumer is cash-constrained, Male should be targeted

#4
df.groupby('Contact Age')['Profit'].sum().plot(kind='bar');
#46 is the target age to maximize profit

#5
df.groupby('Product Type')['Profit'].sum().plot(kind='bar');
#Laptop should be featured

#6
df.groupby('Lead Source')['Profit'].sum().plot(kind='bar');
#Website seems to be the most effective lead source.

#7
df_email = df[df['Lead Source'] == 'Email']
df_email.groupby('Sale Month')['Profit'].sum().plot(kind='bar');
#November is the best time for email marketing.

