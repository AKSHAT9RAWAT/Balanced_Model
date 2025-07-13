import json

# Load the JSON file
with open('caption_groundtruth.json', 'r') as f:
    data = json.load(f)  # Assumes the JSON is a dictionary of objects

# Initialize counters
count_1 = 0
count_0 = 0

# Iterate through each item in the JSON
for item_id, item_data in data.items():
    label = item_data.get('label')  # Extract the label
    if label == 1:
        count_1 += 1
    elif label == 0:
        count_0 += 1

# Print results
print(f"Total '1' (Meaningful): {count_1}")
print(f"Total '0' (Not Meaningful): {count_0}")
print(f"Ratio (1:0): {count_1}:{count_0} or approximately {count_1/(count_1+count_0)*100:.1f}% : {count_0/(count_1+count_0)*100:.1f}%")