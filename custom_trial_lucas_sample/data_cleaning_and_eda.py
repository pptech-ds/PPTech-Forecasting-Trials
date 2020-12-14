import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# %matplotlib inline

import datetime

import statsmodels.tsa.api as smt


def load_data():  
    # return pd.read_csv('data/train.csv')  
    return pd.read_csv('data/lucas_sample.csv')  

sales_data = load_data()

print(sales_data.info())
print(sales_data.head())

def monthly_sales(data):
    monthly_data = data.copy()
    monthly_data.date = monthly_data.date.apply(lambda x: str(x)[:-3])
    monthly_data = monthly_data.groupby('date')['sales'].sum().reset_index()
    monthly_data.date = pd.to_datetime(monthly_data.date)
    return monthly_data

monthly_df = monthly_sales(sales_data)
print(monthly_df.head())

# Duration of dataset
def sales_duration(data):
    data.date = pd.to_datetime(data.date)
    number_of_days = data.date.max() - data.date.min()
    number_of_years = number_of_days.days / 365
    print(number_of_days.days, 'days')
    print(number_of_years, 'years')
    
print(sales_duration(sales_data))

def sales_per_day():
    fig, ax = plt.subplots(figsize=(7,4))
    plt.hist(sales_data.sales, color='mediumblue')
    
    ax.set(xlabel = "Sales Per day",
           ylabel = "Count",
           title = "Distrobution of Sales Per Day")

    plt.savefig('plots/sales_per_day.png')
    
sales_per_day()


# def sales_per_store():
#     by_store = sales_data.groupby('store')['sales'].sum().reset_index()
    
#     fig, ax = plt.subplots(figsize=(7,4))
#     sns.barplot(by_store.store, by_store.sales, color='mediumblue')
    
#     ax.set(xlabel = "Store ID",
#            ylabel = "Number of Sales",
#            title = "Total Sales Per Store")
    
#     sns.despine()

#     plt.savefig('plots/sales_per_store.png')
    
# sales_per_store()


# Average monthly sales

# Overall
avg_monthly_sales = monthly_df.sales.mean()
print(f"Overall average monthly sales: ${avg_monthly_sales}")

# Last 12 months (this will be the forecasted sales)
avg_monthly_sales_12month = monthly_df.sales[-12:].mean()
print(f"Last 12 months average monthly sales: ${avg_monthly_sales_12month}")


def time_plot(data, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(x_col, y_col, data=data, ax=ax, color='mediumblue', label='Total Sales')
    
    second = data.groupby(data.date.dt.year)[y_col].mean().reset_index()
    second.date = pd.to_datetime(second.date, format='%Y')
    sns.lineplot((second.date + datetime.timedelta(6*365/12)), y_col, data=second, ax=ax, color='red', label='Mean Sales')   
    
    ax.set(xlabel = "Date",
           ylabel = "Sales",
           title = title)
    
    sns.despine()

    plt.savefig('plots/time_plot.png')

time_plot(monthly_df, 'date', 'sales', 'Monthly Sales Before Diff Transformation')