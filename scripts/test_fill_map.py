# check whether the filled_map.json file is created correctly.

# Path: datasets/filled_map.json

import json

with open('datasets/filled_map.json') as file:
    filled_dict = json.load(file)

# check if there are keys for every value 0 - 30,522
for i in range(30522):
    assert str(i) in filled_dict, f"Key {i} is missing from the filled dictionary."

# get all the values from the dictionary and check that all values between 0 and 30,522 exist as values
values = set(filled_dict.values())
for i in range(30522):
    assert i in values, f"Value {i} is missing from the filled dictionary."

print(filled_dict['29888'])
print(len(values))