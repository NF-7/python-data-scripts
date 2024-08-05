import pandas as pd
import ast

def clean_serial_numbers(serial_numbers):
    try:
        serial_numbers_list = ast.literal_eval(serial_numbers)
        if isinstance(serial_numbers_list, list):
            return ', '.join(serial_numbers_list)
        else:
            return serial_numbers
    except (ValueError, SyntaxError):
        return serial_numbers

# Load the CSV file
df = pd.read_csv('fullExport.csv', delimiter=';', on_bad_lines='skip')
# Load the CSV file with emails to exclude
exclude_emails_df = pd.read_csv('zendesk_outreach_emails.csv')

print("Column names:", df.columns)
df.columns = df.columns.str.strip()

# Check if the required columns exist
required_columns = ['email', 'SerialNumbers', 'last_session']
if all(col in df.columns for col in required_columns):
    # Filter out rows with empty 'email' or 'SerialNumbers' fields
    df = df.dropna(subset=['email', 'SerialNumbers'])
    # Filter out rows where 'email' contains 'mybirdbuddy' or 'seacomp'
    df = df[~df['email'].str.contains('mybirdbuddy|seacomp', case=False)]
    # Exclude rows where email matches any email in excludeEmails.csv
    exclude_emails_list = exclude_emails_df['email'].tolist()
    df = df[~df['email'].isin(exclude_emails_list)]
    # Convert 'last_session' to datetime format
    df['last_session'] = pd.to_datetime(df['last_session'], errors='coerce')
    # Drop rows where 'last_session' could not be converted to datetime
    df = df.dropna(subset=['last_session'])
    # Convert 'last_session' to date format only
    df['last_session'] = df['last_session'].dt.date
    # Clean up the SerialNumbers column
    df['SerialNumbers'] = df['SerialNumbers'].apply(clean_serial_numbers)
    # Sort the DataFrame by 'last_session' in descending order
    df = df.sort_values(by='last_session', ascending=False)
    # Save the filtered and sorted DataFrame to a new CSV file
    df.to_csv('filtered_sorted_file_matched.csv', index=False)

    print("Filtered and sorted data saved to 'filtered_sorted_file_matched.csv'")
else:
    print("The required columns 'email', 'SerialNumbers', or 'last_session' are not present in the CSV file.")

# IMPROVED TO MATCH AMAZON DEVICES

# import pandas as pd
# import ast

# def clean_serial_numbers(serial_numbers):
#     try:
#         serial_numbers_list = ast.literal_eval(serial_numbers)
#         if isinstance(serial_numbers_list, list):
#             cleaned_serials = []
#             for sn in serial_numbers_list:
#                 # Ensure each serial number has 11 digits by padding with leading zeros
#                 cleaned_serials.append(sn.zfill(11))
#             return ', '.join(cleaned_serials)
#         else:
#             return serial_numbers.zfill(11)
#     except (ValueError, SyntaxError):
#         return serial_numbers

# # Load the CSV file
# df = pd.read_csv('fullExport.csv', delimiter=';', on_bad_lines='skip')
# # Load the CSV file with emails to exclude
# exclude_emails_df = pd.read_csv('zendesk_outreach_emails.csv')
# # Load the CSV file with serial numbers to cross-reference
# serial_numbers_df = pd.read_csv('serialNumbersAmazon.csv')

# print("Column names:", df.columns)
# df.columns = df.columns.str.strip()

# # Ensure required columns are present
# required_columns = ['email', 'SerialNumbers', 'last_session']
# if all(col in df.columns for col in required_columns):
#     # Filter out rows with empty 'email' or 'SerialNumbers' fields
#     df = df.dropna(subset=['email', 'SerialNumbers'])
#     # Filter out rows where 'email' contains 'mybirdbuddy' or 'seacomp'
#     df = df[~df['email'].str.contains('mybirdbuddy|seacomp', case=False)]
#     # Exclude rows where email matches any email in excludeEmails.csv
#     exclude_emails_list = exclude_emails_df['email'].tolist()
#     df = df[~df['email'].isin(exclude_emails_list)]
#     # Convert 'last_session' to datetime format
#     df['last_session'] = pd.to_datetime(df['last_session'], errors='coerce')
#     # Drop rows where 'last_session' could not be converted to datetime
#     df = df.dropna(subset=['last_session'])
#     # Convert 'last_session' to date format only
#     df['last_session'] = df['last_session'].dt.date
#     # Clean up the SerialNumbers column
#     df['SerialNumbers'] = df['SerialNumbers'].apply(clean_serial_numbers)

#     # Print cleaned serial numbers for verification
#     print("Cleaned SerialNumbers:", df['SerialNumbers'].head())

#     # Cross-reference serial numbers and add match column
#     serial_numbers_list = serial_numbers_df['SerialNumbers'].tolist()
#     df['SerialMatch'] = df['SerialNumbers'].apply(lambda x: 'YES' if any(sn in serial_numbers_list for sn in x.split(', ')) else 'NO')

#     # Sort the DataFrame by 'last_session' in descending order
#     df = df.sort_values(by='last_session', ascending=False)
    
#     # Ensure SerialNumbers column is saved as strings
#     df['SerialNumbers'] = df['SerialNumbers'].astype(str)
    
#     # Save the filtered and sorted DataFrame to a new CSV file
#     df.to_csv('filtered_sorted_file_matched.csv', index=False)

#     print("Filtered and sorted data saved to 'filtered_sorted_file_matched.csv'")
# else:
#     print("The required columns 'email', 'SerialNumbers', or 'last_session' are not present in the CSV file.")