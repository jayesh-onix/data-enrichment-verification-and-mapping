import pandas as pd

# Input file
input_file = "onix_enriched_data.csv"

# Output file
output_file = "onix_enriched_data_trimmed_100.csv"

# Load the dataset
df = pd.read_csv(input_file)

# Select first 20 rows
trimmed_df = df.head(20)

# Save the trimmed dataset
trimmed_df.to_csv(output_file, index=False)

print("Dataset trimmed successfully!")
print("Original rows:", len(df))
print("Trimmed rows:", len(trimmed_df))