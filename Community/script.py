import pandas as pd

# Load the data
file_path = 'ProfilePerformance.csv' if True else "PostPerformance.csv"
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime without specifying a format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Check for any rows where the date conversion failed
if data['Date'].isnull().any():
    print("Some dates failed to parse. Check the date formats in your CSV.")

# Normalize the dates by removing the time component
data['Date'] = data['Date'].dt.normalize()

# Convert all columns from the third column onward to numeric, removing commas and forcing non-convertible values to NaN
for col in data.columns[2:]:
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')

# Define aggregation functions for each column, ensuring they are summed
aggregation_functions = {col: 'sum' for col in data.columns[2:]}

# Group by the normalized 'Date' and apply the aggregation functions
aggregated_data = data.groupby('Date').agg(aggregation_functions)

# Save the aggregated data back to CSV
aggregated_data.to_csv('aggregated_data.csv')

print("Aggregation completed and saved to 'aggregated_data.csv'.")
