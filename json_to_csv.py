import json
import csv

def json_to_csv(json_file, csv_file):
    # Read JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure data is a list
    if not isinstance(data, list):
        raise ValueError("JSON data should be an array of objects")

    # Extract keys (column names)
    if not data:
        raise ValueError("JSON array is empty")

    keys = data[0].keys()

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

# Example usage
json_to_csv('data.json', 'data.csv')