import argparse
import json
import pandas as pd
import os

parser = argparse.ArgumentParser(description="Process json results file from one or more directories.")
parser.add_argument("--root-dir", nargs='+', type=str, default=["./logs/"], help="HuggingFace model name or path.")

args = parser.parse_args()

# Provide a list of root directories
root_directories = args.root_dir
if isinstance(root_directories, str):
    root_directories = [root_directories]

# Initialize an empty list to collect directories containing 'results.json'
directories = []

# Populate the directories list with paths to 'results.json' from all root directories
for root_directory in root_directories:
    directories.extend([
        os.path.join(dp, f) for dp, dn, filenames in os.walk(root_directory) for f in filenames if f == 'results.json'
    ])

# Prepare a list to hold all rows of data
all_data = []

# Define the specific subfields to extract
desired_subfields = ["acc,none", "word_perplexity,none", "exact,none", "exact_match,get-answer", "exact_match,remove_whitespace", "exact_match,strict-match", "pass@1"]

# Iterate over each directory
for json_path in directories:
    # Load the JSON file
    with open(json_path, 'r') as file:
        try:
            data = json.load(file)
        except:
            print(f"Failed to parse {json_path}")
            continue

    # Support different variants
    if "results" in data:
        data = data["results"]

    # Prepare a dictionary to hold the extracted data for this file
    extracted_data = {'Directory': os.path.dirname(json_path)}
    
    # Iterate over each key in the JSON data
    for dataset, metrics in data.items():
        if isinstance(metrics, dict):
            # Iterate over each metric in the dataset
            for metric, value in metrics.items():
                # Check if the metric is one of the desired subfields
                if metric in desired_subfields:
                    if 'alias' in metrics:
                        # Format the header as "<alias>,<metric_name_without_none>"
                        header = f"{metrics['alias']},{metric.split(',')[0]}"
                    else:
                        header = f"{dataset},{metric}"
                    extracted_data[header] = value
        else:
            extracted_data[dataset] = metrics
    
    # Append the extracted data to the list
    all_data.append(extracted_data)

# Create a DataFrame from the list of all data
df = pd.DataFrame(all_data)

# Display the DataFrame
print(df)

# Write csv
out_file = "all_results.csv"
print(f"Writing output to {out_file}")
df.to_csv(out_file, index=False)