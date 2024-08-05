import csv
from datetime import datetime

# Define input and output file paths
input_file = 'input.csv'
output_file = 'output.csv'

# Initialize a dictionary to store the highest metric for each date
date_metrics = {}

# Read the input CSV file
with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Combine the row into a single string
        row_str = ','.join(row)
        try:
            # Assume the date and metric are separated by a comma in the single column
            datetime_str, metric_str = row_str.split(',')
            date_str = datetime_str.split(' ')[0]
            date = datetime.strptime(date_str, '%Y/%m/%d').date()  # Adjust date format if necessary
            metric = float(metric_str)
            
            # Update the highest metric for the date
            if date not in date_metrics or metric > date_metrics[date]:
                date_metrics[date] = metric
        except ValueError as e:
            print(f"Skipping row due to error: {row} - {e}")

# Write the results to the output CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Highest Metric'])
    for date, metric in sorted(date_metrics.items()):
        writer.writerow([date, metric])

print(f'Results have been written to {output_file}')