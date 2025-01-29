# Exploratory Data Analysis (EDA) Project

## Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Step 2: Load the Data
data_path = "Ecommerce_data.csv"
df = pd.read_csv(data_path)

# Step 3: Data Overview
print("Dataset Information:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nUnique Values per Column:\n")
print(df.nunique())

print("\nDuplicate Rows:", df.duplicated().sum())

# Step 4: Handle Missing and Duplicate Data
# Drop duplicates
df = df.drop_duplicates()

# Fill missing values
for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=['number']):
    df[col].fillna(df[col].mean(), inplace=True)

# Step 5: Feature Engineering
# Convert date columns to datetime format
date_columns = ['order_date', 'ship_date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=date_columns)

# Create new features

df['actual_shipping_delay'] = (df['ship_date'] - df['order_date']).dt.days
df['profit_margin'] = df['profit_per_order'] / df['sales_per_order']
df['order_year'] = df['order_date'].dt.year
df['order_month'] = df['order_date'].dt.month
df['order_weekday'] = df['order_date'].dt.day_name()
df['returning_customer'] = df.duplicated(subset=['customer_id'], keep=False)
df['shipping_category'] = pd.cut(df['actual_shipping_delay'], bins=[-1, 0, 3, 7, 30],
                                 labels=['Same Day', 'Fast', 'Moderate', 'Delayed'])


# Sales by region
sales_by_region = df.groupby('customer_region')['sales_per_order'].sum().sort_values(ascending=False)
print("\nSales by Region:\n", sales_by_region)

# Sales by product category
sales_by_category = df.groupby('category_name')['sales_per_order'].sum().sort_values(ascending=False)
print("\nSales by Category:\n", sales_by_category)

# Top-performing products
top_products = df.groupby('product_name')['sales_per_order'].sum().sort_values(ascending=False).head(10)
print("\nTop-Performing Products:\n", top_products)

# Monthly revenue
monthly_revenue = df.groupby(['order_year', 'order_month'])['sales_per_order'].sum().reset_index()
print("\nMonthly Revenue:\n", monthly_revenue)

# Step 6: Univariate Analysis
numerical_cols = ['sales_per_order', 'order_quantity', 'profit_per_order', 'actual_shipping_delay']
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

categorical_cols = ['category_name', 'customer_segment', 'delivery_status', 'shipping_type']
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    value_counts = df[col].value_counts()
    plt.bar(value_counts.index, value_counts.values, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Step 7: Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_region.index, y=sales_by_region.values, palette="viridis")
plt.title('Sales by Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='category_name', y='sales_per_order', data=df, ci=None)
plt.title("Sales by Category")
plt.xticks(rotation=45)
plt.show()


# Step 8: Customer Segment Wise Top Sales
customer_segment_sales = df.groupby('customer_segment')['sales_per_order'].sum().sort_values(ascending=False)
print("\nCustomer Segment Wise Top Sales:\n", customer_segment_sales)

plt.figure(figsize=(10, 6))
sns.barplot(x=customer_segment_sales.index, y=customer_segment_sales.values, palette="coolwarm")
plt.title("Top Sales by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

#  Multivariate Analysis
numerical_cols_filtered = [col for col in numerical_cols if col in df.columns]
sns.pairplot(df[numerical_cols_filtered])
plt.show()

#sales per customer segment
plt.figure(figsize=(10, 6))
sns.boxplot(x='customer_segment', y='sales_per_order', data=df)
plt.title("Sales by Customer Segment")
plt.show()

# Time-Based Analysis
df['order_month'] = df['order_date'].dt.to_period('M')
sales_trends = df.groupby('order_month')['sales_per_order'].sum()

plt.figure(figsize=(12, 6))
sales_trends.plot()
plt.title("Sales Trend Over Time")
plt.xlabel("Order Month")
plt.ylabel("Total Sales")
plt.grid()
plt.show()

# Customer Behavior Analysis
customer_spend = df.groupby('customer_id')['sales_per_order'].sum()
print("\nTop 10 Customers by Spending:\n", customer_spend.sort_values(ascending=False).head(10))

#Reporting Insights
print("\nTop Insights:")
print("1. Top performing categories:\n", sales_by_category)
print("\n2. Average profit margin by shipping type:\n", df.groupby('shipping_type')['profit_margin'].mean())
print("\n3. Correlation between sales and profit:\n", df[['sales_per_order', 'profit_per_order']].corr())
print("3. Year-over-Year Sales Growth:\n", df.groupby('order_year')['sales_per_order'].sum().pct_change())

return_rate = df.groupby('category_name')['returning_customer'].mean()
print("Return Rate by Category:\n", return_rate)

customer_segment_sales = df.groupby('customer_segment')['sales_per_order'].sum().sort_values(ascending=False)
print("Customer Segment Wise Top Sales:\n", customer_segment_sales)

#Shipping Delay vs Profit Margin
shipping_delay_profit = df.groupby('shipping_category')['profit_margin'].mean()
print("Profit Margin by Shipping Delay Category:\n", shipping_delay_profit)

#Monthly Sales Trend
monthly_sales = df.groupby(['order_year', 'order_month'])['sales_per_order'].sum().reset_index()
print("Monthly Sales Trend:\n", monthly_sales)

#Product Performance Comparison
product_sales = df.groupby('product_name')['sales_per_order'].sum().sort_values(ascending=False)
print("Product Sales Performance:\n", product_sales)

# Save Cleaned Data
df.to_csv("cleaned_data.csv", index=False)
