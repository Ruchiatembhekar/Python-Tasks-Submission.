import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('sales_data.csv')

print("=== DATA ANALYSIS REPORT ===")
print("\n1. Dataset Overview:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")

print("\n2. Basic Statistics:")
print(df.describe())

# Calculate averages
avg_sales = df['Sales'].mean()
avg_price = df['Price'].mean()

print(f"\n3. Key Insights:")
print(f"Average Sales: {avg_sales:.2f} units")
print(f"Average Price: ${avg_price:.2f}")

# Create visualizations
plt.figure(figsize=(15, 10))

# Bar chart - Average sales by product
plt.subplot(2, 3, 1)
product_sales = df.groupby('Product')['Sales'].mean()
product_sales.plot(kind='bar')
plt.title('Average Sales by Product')
plt.xticks(rotation=45)

# Scatter plot - Price vs Sales
plt.subplot(2, 3, 2)
plt.scatter(df['Price'], df['Sales'], alpha=0.6)
plt.xlabel('Price ($)')
plt.ylabel('Sales (units)')
plt.title('Price vs Sales Correlation')

# Heatmap - Correlation matrix
plt.subplot(2, 3, 3)
correlation_data = df[['Sales', 'Price']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')

# Monthly sales trend
plt.subplot(2, 3, 4)
monthly_sales = df.groupby('Month')['Sales'].sum()
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45)

# Product distribution pie chart
plt.subplot(2, 3, 5)
product_counts = df['Product'].value_counts()
plt.pie(product_counts.values, labels=product_counts.index, autopct='%1.1f%%')
plt.title('Product Distribution')

plt.tight_layout()
plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n4. Analysis Complete!")
print("Visualizations saved as 'analysis_results.png'") 
