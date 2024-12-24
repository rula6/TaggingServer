import requests

url = "http://127.0.0.1:8000/upload-file/"
file_path = "gojo.jpg"

with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": f})

if response.status_code == 200:
    print("Predictions:", response.json())
else:
    print("Error:", response.text)
