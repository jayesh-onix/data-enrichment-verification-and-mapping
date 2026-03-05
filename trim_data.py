import pandas as pd

# Input file
input_file = "/home/jayesh/onix/internal_projects/data_enrichment_and_clubing/onix_enriched_data.csv"

# Output file
output_file = "/home/jayesh/onix/internal_projects/data_enrichment_and_clubing/onix_enriched_data_trimmed.csv"

# Load the dataset
df = pd.read_csv(input_file)

# Select first 5000 rows
trimmed_df = df.head(3000)

# Save the trimmed dataset
trimmed_df.to_csv(output_file, index=False)

print("Dataset trimmed successfully!")
print("Original rows:", len(df))
print("Trimmed rows:", len(trimmed_df))