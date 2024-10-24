import pandas as pd

# Load the CSV file into a DataFrame
input_file = 'Covid_Dataset'  # Replace with your input file path
output_file = 'output.csv'  # Replace with your desired output file path
# Read the CSV file
df = pd.read_csv(input_file)

# Specify the columns to be deleted
columns_to_delete = ['Wearing Masks','Sanitization from Market']  # Replace with the names of the columns you want to delete

# Drop the specified columns
df_new = df.drop(columns=columns_to_delete)

# Save the resulting DataFrame to a new CSV file
df_new.to_csv(output_file, index=False)

print(f"New CSV file created without {columns_to_delete}: {output_file}")