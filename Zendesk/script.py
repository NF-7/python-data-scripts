import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

def process_tickets(file_path):
    # Load the data
    df = pd.read_csv(file_path, delimiter=';')
    print(f"Data loaded with {len(df)} entries.")

    # Convert 'Created at' and 'Solved at' to datetime and normalize to 00:00:00
    df['Created at'] = pd.to_datetime(df['Created at']).dt.normalize()
    df['Solved at'] = pd.to_datetime(df['Solved at']).dt.normalize()

    # Update 'Requester wait time in minutes' for unresolved tickets
    today = pd.Timestamp.now().normalize()
    unsolved_mask = df['Solved at'].isna()
    df.loc[unsolved_mask, 'Requester wait time in minutes'] = (today - df['Created at']).dt.total_seconds() / 60

    # Convert 'First reply time in minutes' to hours
    df['First reply time in hours'] = df['First reply time in minutes'] / 60.0

    # Generate the full date range for the DataFrame columns
    min_date = df['Created at'].min()
    max_date = df['Solved at'].max() if df['Solved at'].notna().any() else today  # Use today if some tickets are unsolved
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

    print(f"Date range from {min_date} to {max_date}")

    # Initialize metrics DataFrame
    metrics = [
        'NEW', 'CURRENTLY OPEN', 'SOLVED', 'ON HOLD',
        'AVG RESOLUTION TIME (hours)', 'MEDIAN RESOLUTION TIME (hours)',
        'WORST 10PCT RESOLUTION TIME (hours)', 'TOP 25PCT QUICKEST RESOLUTION TIME (hours)',
        'AVG FIRST REPLY TIME (hours)', 'WORST 10PCT FIRST REPLY TIME (hours)',
        'AVG OPEN TIME (hours)', 'MEDIAN OPEN TIME (hours)',
        'WORST 10PCT OPEN TIME (hours)', 'TOP 25PCT BEST OPEN TIME (hours)',
        'Open Times average (without ON HOLD tickets)', 'Resolution Times average (without ON HOLD tickets)'
    ]
    output_df = pd.DataFrame(index=metrics, columns=all_dates)
    output_df.fillna(0.0, inplace=True)

    # Populate 'NEW' and 'SOLVED' status
    new_counts = df['Created at'].dt.floor('D')
    solved_counts = df['Solved at'].dropna().dt.floor('D')

    # Count occurrences and reindex to match all_dates
    new_counts = new_counts.value_counts().reindex(all_dates, fill_value=0)
    solved_counts = solved_counts.value_counts().reindex(all_dates, fill_value=0)

    output_df.loc['NEW', new_counts.index] = new_counts.values
    output_df.loc['SOLVED', solved_counts.index] = solved_counts.values

    print("New counts summary:", new_counts.sum())
    print("Solved counts summary:", solved_counts.sum())

    # Track 'ON HOLD' tickets correctly
    on_hold_status = pd.Series(0, index=all_dates)

    for date in all_dates:
        on_hold_tickets = df[(df['Created at'] <= date) & ((df['Solved at'] > date) | (df['Solved at'].isna())) & (df['On hold time in minutes'] > 0)]
        on_hold_status[date] = on_hold_tickets.shape[0]

    output_df.loc['ON HOLD', on_hold_status.index] = on_hold_status.values

    # Populate 'CURRENTLY OPEN' status
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing tickets"):
        start_date = row['Created at']
        end_date = row['Solved at'] if pd.notna(row['Solved at']) else all_dates[-1]
        open_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        output_df.loc['CURRENTLY OPEN', open_dates] += 1

    # Calculate dynamic metrics for each date
    avg_res_time = []
    median_res_time = []
    worst_10_res_time = []
    top_25p_res_time = []
    avg_first_reply_time = []
    worst_10_first_reply_time = []
    avg_open_time = []
    median_open_time = []
    worst_10_open_time = []
    top_25p_best_open_time = []
    avg_open_time_wo_on_hold = []
    avg_res_time_wo_on_hold = []

    for date in all_dates:
        open_tickets = df[(df['Created at'] <= date) & ((df['Solved at'] > date) | (df['Solved at'].isna()))]
        resolved_tickets = df[df['Solved at'] == date]
        created_tickets = df[df['Created at'] == date]

        # Calculate resolution time for resolved tickets
        resolved_times = (resolved_tickets['Solved at'] - resolved_tickets['Created at']).dt.total_seconds() / 3600  # Convert to hours
        resolved_times_wo_on_hold = resolved_times[resolved_tickets['On hold time in minutes'] == 0]

        # Filter out tickets with zero resolution time
        resolved_times = resolved_times[resolved_times > 0]
        resolved_times_wo_on_hold = resolved_times_wo_on_hold[resolved_times_wo_on_hold > 0]

        if not resolved_times.empty:
            avg_res_time.append(resolved_times.mean())
            median_res_time.append(resolved_times.median())
            worst_10_res_time.append(resolved_times.quantile(0.90))
            top_25p_res_time.append(resolved_times.quantile(0.25))
        else:
            avg_res_time.append(0.0)
            median_res_time.append(0.0)
            worst_10_res_time.append(0.0)
            top_25p_res_time.append(0.0)

        if not resolved_times_wo_on_hold.empty:
            avg_res_time_wo_on_hold.append(resolved_times_wo_on_hold.mean())
        else:
            avg_res_time_wo_on_hold.append(0.0)

        # Calculate first reply time metrics
        first_reply_times = created_tickets['First reply time in hours'].dropna()

        if not first_reply_times.empty:
            avg_first_reply_time.append(first_reply_times.mean())
            worst_10_first_reply_time.append(first_reply_times.quantile(0.90))
        else:
            avg_first_reply_time.append(0.0)
            worst_10_first_reply_time.append(0.0)

        # Calculate average open time for tickets that were open on this specific day
        open_times = (date - open_tickets['Created at']).dt.total_seconds() / 3600  # Convert to hours
        open_times_wo_on_hold = open_times[open_tickets['On hold time in minutes'] == 0]

        if not open_times.empty:
            avg_open_time.append(open_times.mean())
            median_open_time.append(open_times.median())
            worst_10_open_time.append(open_times.quantile(0.90))
            top_25p_best_open_time.append(open_times.quantile(0.25))
        else:
            avg_open_time.append(0.0)
            median_open_time.append(0.0)
            worst_10_open_time.append(0.0)
            top_25p_best_open_time.append(0.0)

        if not open_times_wo_on_hold.empty:
            avg_open_time_wo_on_hold.append(open_times_wo_on_hold.mean())
        else:
            avg_open_time_wo_on_hold.append(0.0)

    avg_res_time_series = pd.Series(avg_res_time, index=all_dates)
    median_res_time_series = pd.Series(median_res_time, index=all_dates)
    worst_10_res_time_series = pd.Series(worst_10_res_time, index=all_dates)
    top_25p_res_time_series = pd.Series(top_25p_res_time, index=all_dates)
    avg_first_reply_time_series = pd.Series(avg_first_reply_time, index=all_dates)
    worst_10_first_reply_time_series = pd.Series(worst_10_first_reply_time, index=all_dates)
    avg_open_time_series = pd.Series(avg_open_time, index=all_dates)
    median_open_time_series = pd.Series(median_open_time, index=all_dates)
    worst_10_open_time_series = pd.Series(worst_10_open_time, index=all_dates)
    top_25p_best_open_time_series = pd.Series(top_25p_best_open_time, index=all_dates)
    avg_open_time_wo_on_hold_series = pd.Series(avg_open_time_wo_on_hold, index=all_dates)
    avg_res_time_wo_on_hold_series = pd.Series(avg_res_time_wo_on_hold, index=all_dates)

    output_df.loc['AVG RESOLUTION TIME (hours)', :] = avg_res_time_series
    output_df.loc['MEDIAN RESOLUTION TIME (hours)', :] = median_res_time_series
    output_df.loc['WORST 10PCT RESOLUTION TIME (hours)', :] = worst_10_res_time_series
    output_df.loc['TOP 25PCT QUICKEST RESOLUTION TIME (hours)', :] = top_25p_res_time_series
    output_df.loc['AVG FIRST REPLY TIME (hours)', :] = avg_first_reply_time_series
    output_df.loc['WORST 10PCT FIRST REPLY TIME (hours)', :] = worst_10_first_reply_time_series
    output_df.loc['AVG OPEN TIME (hours)', :] = avg_open_time_series
    output_df.loc['MEDIAN OPEN TIME (hours)', :] = median_open_time_series
    output_df.loc['WORST 10PCT OPEN TIME (hours)', :] = worst_10_open_time_series
    output_df.loc['TOP 25PCT BEST OPEN TIME (hours)', :] = top_25p_best_open_time_series
    output_df.loc['Open Times average (without ON HOLD tickets)', :] = avg_open_time_wo_on_hold_series
    output_df.loc['Resolution Times average (without ON HOLD tickets)', :] = avg_res_time_wo_on_hold_series

    # Debugging: Print summary statistics
    print("AVG RESOLUTION TIME (hours):", avg_res_time_series.describe())
    print("MEDIAN RESOLUTION TIME (hours):", median_res_time_series.describe())
    print("WORST 10PCT RESOLUTION TIME (hours):", worst_10_res_time_series.describe())
    print("TOP 25PCT QUICKEST RESOLUTION TIME (hours):", top_25p_res_time_series.describe())
    print("AVG FIRST REPLY TIME (hours):", avg_first_reply_time_series.describe())
    print("MEDIAN OPEN TIME (hours):", median_open_time_series.describe())
    print("WORST 10PCT OPEN TIME (hours):", worst_10_open_time_series.describe())
    print("TOP 25PCT BEST OPEN TIME (hours):", top_25p_best_open_time_series.describe())

    # Calculate monthly average resolution time
    df['Month'] = df['Solved at'].dt.to_period('M')
    monthly_avg_res_time = df.dropna(subset=['Solved at']).groupby('Month').apply(
        lambda x: (x['Solved at'] - x['Created at']).dt.total_seconds().mean() / 3600
    )

    # Calculate monthly average resolution time for tickets without on-hold tickets
    monthly_avg_res_time_wo_on_hold = df[(df['Solved at'].notna()) & (df['On hold time in minutes'] == 0)].groupby('Month').apply(
        lambda x: (x['Solved at'] - x['Created at']).dt.total_seconds().mean() / 3600
    )

    # Print Monthly Average Resolution Time
    print("Monthly Average Resolution Time (hours):")
    print(monthly_avg_res_time)

    print("Monthly Average Resolution Time (without ON HOLD tickets) (hours):")
    print(monthly_avg_res_time_wo_on_hold)

    # Add Monthly Average Resolution Time to output_df
    monthly_avg_res_time_df = monthly_avg_res_time.to_frame().T
    monthly_avg_res_time_df.index = ['MONTHLY AVG RESOLUTION TIME (hours)']

    monthly_avg_res_time_wo_on_hold_df = monthly_avg_res_time_wo_on_hold.to_frame().T
    monthly_avg_res_time_wo_on_hold_df.index = ['MONTHLY AVG RESOLUTION TIME (without ON HOLD tickets) (hours)']

    # Ensure the dates in monthly_avg_res_time_df are the end of each month in all_dates
    monthly_avg_res_time_df.columns = [pd.Period(col, freq='M').end_time for col in monthly_avg_res_time_df.columns]
    monthly_avg_res_time_wo_on_hold_df.columns = [pd.Period(col, freq='M').end_time for col in monthly_avg_res_time_wo_on_hold_df.columns]

    # Merge the data into output_df
    output_df = pd.concat([output_df, monthly_avg_res_time_df, monthly_avg_res_time_wo_on_hold_df], axis=0)

    return output_df, monthly_avg_res_time, monthly_avg_res_time_wo_on_hold

# Replace 'file_path' with the path to your CSV file
file_path = "tickets.csv"
df_processed, monthly_avg_res_time, monthly_avg_res_time_wo_on_hold = process_tickets(file_path)  # Unpack the returned tuple

# Get the current date
current_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Create the filename with the current date
filename = f'output_tickets-full_{current_date}.csv'

# Save the processed data to a new CSV
df_processed.to_csv(filename)
monthly_avg_res_time.to_csv('monthly_avg_resolution_time.csv')
monthly_avg_res_time_wo_on_hold.to_csv('monthly_avg_resolution_time_wo_on_hold.csv')

# ## VISUALIZATIONS

# # Visualize difference between NEW and SOLVED
# Load your CSV file
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
file_path = f'output_tickets-full_{current_date}.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Process the dataframe
df.set_index(df.columns[0], inplace=True)
df.columns = pd.to_datetime(df.columns)

# Extract relevant series
new_tickets = df.loc['NEW']
resolved_tickets = df.loc['SOLVED']
on_hold_tickets = df.loc['ON HOLD']

# Convert indices to datetime
new_tickets.index = pd.to_datetime(new_tickets.index)
resolved_tickets.index = pd.to_datetime(resolved_tickets.index)
on_hold_tickets.index = pd.to_datetime(on_hold_tickets.index)

# Aggregate the data by week
weekly_new_tickets = new_tickets.resample('W').sum()
weekly_resolved_tickets = resolved_tickets.resample('W').sum()
weekly_on_hold_tickets = on_hold_tickets.resample('W').sum()

# Plot new, resolved, and on-hold tickets over time (weekly data)
plt.figure(figsize=(14, 7))
plt.plot(weekly_new_tickets.index, weekly_new_tickets.values, label='New Tickets', color='blue')
plt.plot(weekly_resolved_tickets.index, weekly_resolved_tickets.values, label='Resolved Tickets', color='green')
plt.plot(weekly_on_hold_tickets.index, weekly_on_hold_tickets.values, label='On Hold Tickets', color='orange')
plt.xlabel('Date')
plt.ylabel('Number of Tickets')
plt.title('Weekly New, Resolved, and On Hold Tickets Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Visualize resolution time metrics
avg_res_time = df.loc['AVG RESOLUTION TIME (hours)']
worst_10pct_res_time = df.loc['WORST 10PCT RESOLUTION TIME (hours)']
top_25pct_quickest_res_time = df.loc['TOP 25PCT QUICKEST RESOLUTION TIME (hours)']
avg_res_time_wo_on_hold = df.loc['Resolution Times average (without ON HOLD tickets)']

# Convert indices to datetime
avg_res_time.index = pd.to_datetime(avg_res_time.index)
worst_10pct_res_time.index = pd.to_datetime(worst_10pct_res_time.index)
top_25pct_quickest_res_time.index = pd.to_datetime(top_25pct_quickest_res_time.index)
avg_res_time_wo_on_hold.index = pd.to_datetime(avg_res_time_wo_on_hold.index)

# Plot resolution time metrics over time
plt.figure(figsize=(14, 7))
plt.plot(avg_res_time.index, avg_res_time.values, label='Average Resolution Time (hours)', color='blue')
plt.plot(worst_10pct_res_time.index, worst_10pct_res_time.values, label='Worst 10% Resolution Time (hours)', color='red')
plt.plot(top_25pct_quickest_res_time.index, top_25pct_quickest_res_time.values, label='Top 25% Quickest Resolution Time (hours)', color='green')
plt.plot(avg_res_time_wo_on_hold.index, avg_res_time_wo_on_hold.values, label='Resolution Times avg (without ON HOLD)', color='purple', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Resolution Time (hours)')
plt.title('Resolution Time Metrics Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Visualize open time metrics
avg_open_time = df.loc['AVG OPEN TIME (hours)']
avg_open_time_wo_on_hold = df.loc['Open Times average (without ON HOLD tickets)']

# Convert indices to datetime
avg_open_time.index = pd.to_datetime(avg_open_time.index)
avg_open_time_wo_on_hold.index = pd.to_datetime(avg_open_time_wo_on_hold.index)

# Plot open time metrics over time
plt.figure(figsize=(14, 7))
plt.plot(avg_open_time.index, avg_open_time.values, label='Average Open Time (hours)', color='blue')
plt.plot(avg_open_time_wo_on_hold.index, avg_open_time_wo_on_hold.values, label='Open Times avg (without ON HOLD)', color='purple', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Open Time (hours)')
plt.title('Open Time Metrics Over Time')
plt.legend()
plt.grid(True)
plt.show()