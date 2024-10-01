from pyspark.sql import functions as F
from pyspark.sql import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

def eda_report_spark(df: DataFrame):
    # 1. Basic Information
    print("Basic Information")
    print("-----------------")
    df.printSchema()
    print(f"Total Rows: {df.count()}")
    print("\n")

    # 2. Descriptive Statistics for Numerical Columns
    print("Descriptive Statistics for Numerical Columns")
    print("--------------------------------------------")
    numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (F.IntegerType, F.DoubleType, F.FloatType, F.LongType))]
    summary = df.describe(numeric_cols)
    summary.show()

    # 3. Null Values in the Dataset
    print("Missing Values")
    print("--------------")
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
    total_rows = df.count()
    missing_data = {col: (null_counts[col], (null_counts[col] / total_rows) * 100) for col in null_counts if null_counts[col] > 0}
    
    if missing_data:
        for col, (miss_count, miss_percent) in missing_data.items():
            print(f"{col}: {miss_count} missing values ({miss_percent:.2f}%)")
    else:
        print("No missing values found.")
    print("\n")

    # 4. Distribution of Numerical Columns
    print("Distribution of Numerical Columns")
    print("---------------------------------")
    for col in numeric_cols:
        pandas_df = df.select(col).toPandas()
        plt.figure(figsize=(6, 4))
        sns.histplot(pandas_df[col], bins=15, kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    # 5. Checking for Outliers using Boxplots
    print("Outliers in Numerical Columns")
    print("-----------------------------")
    for col in numeric_cols:
        pandas_df = df.select(col).toPandas()
        plt.figure(figsize=(6, 2))
        sns.boxplot(x=pandas_df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

    # 6. Correlation and Collinearity
    print("Correlation Matrix for Numerical Columns")
    print("----------------------------------------")
    corr_matrix = df.select(numeric_cols).toPandas().corr()
    print(corr_matrix)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

    # Identify collinear pairs
    threshold = 0.8  # High collinearity threshold
    collinear_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                collinear_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    
    if collinear_pairs:
        print("Highly Collinear Pairs (Threshold > 0.8)")
        print("---------------------------------------")
        for pair in collinear_pairs:
            print(f"{pair[0]} and {pair[1]}: Correlation = {corr_matrix.loc[pair[0], pair[1]]:.3f}")
    else:
        print("No highly collinear pairs found with a threshold greater than 0.8.")
    
    print("\n")

    # 7. Value Counts for Categorical Columns
    print("Value Counts for Categorical Columns")
    print("------------------------------------")
    cat_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, F.StringType)]
    
    for col in cat_cols:
        print(f"Value counts for {col}:")
        df.groupBy(col).count().orderBy(F.desc("count")).show(10, False)
    
    print("\n")

    # 8. Unique Values in Each Column
    print("Unique Values in Each Column")
    print("----------------------------")
    for col in df.columns:
        unique_count = df.select(col).distinct().count()
        print(f"{col} has {unique_count} unique values.")
    print("\n")

# Example Usage
# Load a dataset as a PySpark DataFrame (df)
# df = spark.read.csv("your_data.csv", header=True, inferSchema=True)
# eda_report_spark(df)
