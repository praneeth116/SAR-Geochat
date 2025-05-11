# import json

# # Load your JSON file
# with open('annotations.json', 'r') as file:
#     data = json.load(file)  # Assuming the JSON file is a list of dictionaries

# # Iterate through the data and update the "value" field
# for element in data:
#     conversations = element.get("conversations", [])
#     for convo in conversations:
#         if convo.get("from") == "gpt" and isinstance(convo.get("value"), dict):
#             # Convert the value dictionary to a string
#             convo["value"] = json.dumps(convo["value"])

# # Save the modified data back to a JSON file
# with open('updated_data.json', 'w') as file:
#     json.dump(data, file, indent=4)

print("{\"fishing_vessels\": \"\", \"non_fishing_vessels\": \"\", \"non_vessels\": \"{<423><487>}\"}")