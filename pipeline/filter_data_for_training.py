import pandas as pd

# Load the CSV file into a DataFrame
file_path = "../data/tess-disposition-data.csv"  # Replace with your actual file path

# Read the CSV
df = pd.read_csv(file_path)

# Define dispositions for positive and negative classes
positive_labels = ['KP', 'CP']
negative_labels = ['FP', 'FA']

# Filter pipeline
positive_df = df[df['tfopwg_disp'].isin(positive_labels)]
negative_df = df[df['tfopwg_disp'].isin(negative_labels)]

# Save filtered pipeline to separate CSV files
positive_df.to_csv('positive_data.csv', index=False)
negative_df.to_csv('negative_data.csv', index=False)

print("Positive data saved to positive_data.csv")
print("Negative data saved to negative_data.csv")