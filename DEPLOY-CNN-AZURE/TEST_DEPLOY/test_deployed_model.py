import requests
import base64
import json

url = "<your-endpoint>"

aml_token = "<your-token>"


# Create headers with the API key
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {aml_token}"
}

# Read the image
with open('TEST_DEPLOY/00fc2ee5-729f-4757-8aeb-65c3355874f2___RS_HL 1864.JPG', 'rb') as image_file:
    image_data = image_file.read()

# Convert image to base64
image_base64 = base64.b64encode(image_data).decode('utf-8')

# Construct the JSON payload
json_payload = {
    "data": image_base64
}

print(json_payload)

# Send the POST request with the JSON payload
response = requests.post(url, json=json_payload, headers=headers)

# Convert the string to a dictionary
prediction = json.loads(response.json())

# Now dict_data is a Python dictionary
print(prediction)
