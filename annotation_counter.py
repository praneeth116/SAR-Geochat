import json

# Function to count occurrences of the specific value in the JSON data
def count_specific_value(file_path, target_value):
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(len(data))
    # Flattening the JSON data to check all "value" keys
    count = 0
    file_count=0
    for item in data:
        file_count+=1
        if "conversations" in item:
            for conversation in item["conversations"]:
                if conversation.get("value") == target_value:
                    count += 1
    return count,file_count

# Specify the JSON file path and the target value to search for
file_path = '/home/cvpr_ug_4/GeoChat/annotations.json'  # Replace with your actual file path
target_value = "{\"fishing_vessels\": \"\", \"non_fishing_vessels\": \"\", \"non_vessels\": \"\"}"

# Count and print the occurrences
occurrences,x = count_specific_value(file_path, target_value)
print(f"The value '{target_value}' appears {occurrences} times in the JSON file.")
print(x)