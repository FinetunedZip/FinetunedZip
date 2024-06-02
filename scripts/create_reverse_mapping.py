import json

input_file = 'datasets/filled_map.json'
output_file = 'datasets/reverse_filled_mapping.json'

# Load the input JSON file
with open(input_file, 'r') as f:
    data = json.load(f)

# # increment each value in dictionary by 1
# for key in data:
#     data[key] = int(data[key]) + 1

# # save the dictionary as a JSON file
# with open(output_file, 'w') as f:
#     json.dump(data, f)

# Create the reverse dictionary
reverse_dict = {value: int(key) for key, value in data.items()}

# Store the reverse dictionary as a JSON file
with open(output_file, 'w') as f:
    json.dump(reverse_dict, f)
