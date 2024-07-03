import json
import os

data_path = 'data/MATH'  # Adjust this path to your JSON files directory

sample_file = os.path.join(data_path, os.listdir(data_path)[0])
with open(sample_file, 'r') as file:
    sample_data = json.load(file)

print(json.dumps(sample_data, indent=2))
