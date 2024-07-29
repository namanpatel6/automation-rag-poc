import json

import requests
import urllib.request

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    # This will open a browser page for
    credential = InteractiveBrowserCredential()

try:
    ml_client = MLClient.from_config(credential=credential)
except Exception as ex:
    ml_client = MLClient.from_config(credential=credential, path='config.json')

print(ml_client)
DATA_PLANE_TOKEN = ml_client.online_endpoints.get_keys(name="naman-ml-workspace-jckgh").primary_key

url = "https://naman-ml-workspace-jckgh.southcentralus.inference.ml.azure.com/score"
data = {"data": ["This is a sentence to embed.", "Another sentence to embed."]}
headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer { DATA_PLANE_TOKEN }'}
req = urllib.request.Request(url, str.encode(json.dumps(data)), headers)
response = urllib.request.urlopen(req)

result = response.read()
print(result)
# print(response.json())