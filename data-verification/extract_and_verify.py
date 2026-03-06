import csv
import argparse
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default="data/onix_enriched_data.csv")
    parser.add_argument("--filter-file", type=Path, default="data/verification-sample-report-all-false.csv")
    parser.add_argument("--output", type=Path, default="data/filtered_data.csv")
    args = parser.parse_args()

    account_ids = set()
    with open(args.filter_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            account_id = row.get('Account ID 18 Digit')
            if account_id:
                account_ids.add(account_id)

    print(f"Found {len(account_ids)} Account IDs to filter for.")

    filtered_rows = []
    fieldnames = None
    
    with open(args.source, 'r', encoding='utf-8') as f:
        # Skip the first empty line, which was seen in the file preview
        next(f)
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get('Salesforce Account ID') in account_ids:
                filtered_rows.append(row)

    print(f"Extracted {len(filtered_rows)} rows matching the Account IDs.")

    if not filtered_rows:
        print("No matching rows found. Exiting.")
        return

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
        
    print(f"Saved filtered data to {args.output}")

if __name__ == "__main__":
    main()
