import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime

def process_tickets(file_path):
    # Load the data
    df = pd.read_csv(file_path, delimiter=';')
    print(f"Data loaded with {len(df)} entries.")

    # Convert 'Created at' and 'Solved at' to datetime and normalize
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
    all_dates += pd.Timedelta(seconds=86399)  # Adjust to set time to 23:59

    print(f"Date range from {min_date} to {max_date}")

    # Initialize metrics DataFrame
    metrics = [
        'NEW', 'CURRENTLY OPEN', 'SOLVED', 'AVG RESOLUTION TIME (hours)',
        'MEDIAN RESOLUTION TIME (hours)', 'WORST 10PCT RESOLUTION TIME (hours)', 
        'TOP 25PCT QUICKEST RESOLUTION TIME (hours)', 'AVG FIRST REPLY TIME (hours)', 
        'WORST 10PCT FIRST REPLY TIME (hours)', 'AVG OPEN TIME (hours)',
        'MEDIAN OPEN TIME (hours)', 'WORST 10PCT OPEN TIME (hours)', 
        'TOP 25PCT BEST OPEN TIME (hours)'
    ]
    output_df = pd.DataFrame(index=metrics, columns=all_dates)
    output_df.fillna(0.0, inplace=True)

    # Populate 'NEW' and 'SOLVED' status
    new_counts = df['Created at'].dt.floor('D') + pd.Timedelta(seconds=86399)
    solved_counts = df['Solved at'].dropna().dt.floor('D') + pd.Timedelta(seconds=86399)

    # Count occurrences and reindex to match all_dates
    new_counts = new_counts.value_counts().reindex(all_dates, fill_value=0)
    solved_counts = solved_counts.value_counts().reindex(all_dates, fill_value=0)

    output_df.loc['NEW', new_counts.index] = new_counts.values
    output_df.loc['SOLVED', solved_counts.index] = solved_counts.values

    print("New counts summary:", new_counts.sum())
    print("Solved counts summary:", solved_counts.sum())

    # Populate 'CURRENTLY OPEN' status
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing tickets"):
        start_date = row['Created at'] + pd.Timedelta(seconds=86399)
        end_date = (row['Solved at'] + pd.Timedelta(seconds=86399)) if pd.notna(row['Solved at']) else all_dates[-1]
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

    for date in all_dates:
        date = date.normalize()
        open_tickets = df[(df['Created at'] <= date) & ((df['Solved at'] > date) | (df['Solved at'].isna()))]
        resolved_tickets = df[df['Solved at'] == date]
        created_tickets = df[df['Created at'] == date]

        # Calculate resolution time for resolved tickets, discard tickets without 'Solved at'
        resolved_times = (resolved_tickets['Solved at'] - resolved_tickets['Created at']).dt.total_seconds() / 3600  # Convert to hours

        # Filter out tickets with zero resolution time
        resolved_times = resolved_times[resolved_times > 0]

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

        # Calculate first reply time metrics, discard tickets without 'First reply time in hours'
        first_reply_times = created_tickets['First reply time in hours'].dropna()

        if not first_reply_times.empty:
            avg_first_reply_time.append(first_reply_times.mean())
            worst_10_first_reply_time.append(first_reply_times.quantile(0.90))
        else:
            avg_first_reply_time.append(0.0)
            worst_10_first_reply_time.append(0.0)

        # Calculate average open time for tickets that were open on this specific day
        open_times = (date - open_tickets['Created at']).dt.total_seconds() / 3600  # Convert to hours
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

    # Print Monthly Average Resolution Time
    print("Monthly Average Resolution Time (hours):")
    print(monthly_avg_res_time)

    # Add Monthly Average Resolution Time to output_df
    monthly_avg_res_time_df = monthly_avg_res_time.to_frame().T
    monthly_avg_res_time_df.index = ['MONTHLY AVG RESOLUTION TIME (hours)']

    # Ensure the dates in monthly_avg_res_time_df are the end of each month in all_dates
    monthly_avg_res_time_df.columns = [pd.Period(col, freq='M').end_time for col in monthly_avg_res_time_df.columns]

    # Merge the data into output_df
    output_df = pd.concat([output_df, monthly_avg_res_time_df], axis=0)

    return output_df, monthly_avg_res_time

# Replace 'file_path' with the path to your CSV file
file_path = "tickets.csv"
df_processed, monthly_avg_res_time = process_tickets(file_path)  # Unpack the returned tuple

# Get the current date
current_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Calculate the date 6 months ago
six_months_ago = current_date - pd.DateOffset(months=6)

# Filter the DataFrame to include only the last 6 months
filtered_df = df_processed.loc[:, six_months_ago:current_date]

# Create the filename with the current date
filename = f'output_tickets-full_{current_date.strftime("%Y-%m-%d")}.csv'

# Save the processed data to a new CSV
filtered_df.to_csv(filename)
monthly_avg_res_time.to_csv('monthly_avg_resolution_time.csv')