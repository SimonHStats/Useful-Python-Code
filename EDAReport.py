import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def eda_report(df):
    # 1. Basic Information
    print("Basic Information")
    print("-----------------")
    print(df.info())
    print("\n")

    # 2. Descriptive Statistics
    print("Descriptive Statistics")
    print("----------------------")
    print(df.describe(include='all'))
    print("\n")

    # 3. Null Values in the Dataset
    print("Missing Values")
    print("--------------")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
    print(missing_data[missing_data['Missing Values'] > 0])
    print("\n")

    # 4. Distribution of Numerical Columns
    print("Distribution of Numerical Columns")
    print("---------------------------------")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols].hist(bins=15, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    # 5. Check for Outliers using Boxplots
    print("Outliers in Numerical Columns")
    print("-----------------------------")
    for col in num_cols:
        plt.figure(figsize=(5, 2))
        sns.boxplot(df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

    # 6. Correlation and Collinearity
    print("Correlation Matrix")
    print("------------------")
    correlation_matrix = df[num_cols].corr()
    print(correlation_matrix)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

    # Identify pairs with high collinearity
    threshold = 0.8  # Define high collinearity threshold
    collinear_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                collinear_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

    if collinear_pairs:
        print("Highly Collinear Pairs (Threshold > 0.8)")
        print("---------------------------------------")
        for pair in collinear_pairs:
            col1, col2 = pair
            corr_value, _ = pearsonr(df[col1].fillna(0), df[col2].fillna(0))
            print(f"{col1} and {col2}: Pearson correlation = {corr_value:.3f}")
    else:
        print("No highly collinear pairs found with a threshold greater than 0.8.")
    
    print("\n")

    # 7. Value Counts for Categorical Columns
    print("Value Counts for Categorical Columns")
    print("------------------------------------")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"Value counts for {col}:")
        print(df[col].value_counts())
        print("\n")

    # 8. Unique Values in Columns
    print("Unique Values in Each Column")
    print("----------------------------")
    for col in df.columns:
        print(f"{col} has {df[col].nunique()} unique values.")
    print("\n")

# Example Usage
# Load a dataset as a Pandas DataFrame (df)
# df = pd.read_csv("your_data.csv") or pd.DataFrame(your_dataframe)
# eda_report(df)
