import os

# Path to the folder containing the tables
folder_path = "/path/to/your/tables"

# Function to check if a Delta table has the 'risk_owner' column
def check_column_in_delta_table(delta_table_path):
    try:
        # Read the Delta table
        df = spark.read.format("delta").load(delta_table_path)

        # Get the list of columns in the dataset
        columns = df.columns

        # Check if the 'risk_owner' column exists
        if 'risk_owner' in columns:
            print(f"The file {os.path.basename(delta_table_path)} has a column called 'risk_owner'.")
        else:
            print(f"The file {os.path.basename(delta_table_path)} does not have a 'risk_owner' column.")
    
    except Exception as e:
        print(f"Could not process the Delta table {os.path.basename(delta_table_path)}. Error: {e}")

# Iterate over all directories in the folder (each directory is assumed to be a Delta table)
for delta_table in os.listdir(folder_path):
    delta_table_path = os.path.join(folder_path, delta_table)

    # Check if it's a directory (Delta tables are stored as directories)
    if os.path.isdir(delta_table_path):
        check_column_in_delta_table(delta_table_path)


