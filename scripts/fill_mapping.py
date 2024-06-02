import json

# Load the existing dictionary from the JSON file
with open('datasets/data_100MB_mapping_key.json') as file:
    existing_dict = json.load(file)

# Create a set of existing keys
existing_keys = set(existing_dict.keys())

cnt = 1
# Iterate over the range of values from 0 to 30522
for value in range(30522):
    # Check if the value is missing
    if str(value) not in existing_keys:
        # Assign the next value from the queue for missing keys
        existing_dict[str(value)] = cnt + 26403
        cnt += 1

# Save the updated dictionary back to the JSON file
with open('datasets/filled_map.json', 'w') as file:
    json.dump(existing_dict, file)