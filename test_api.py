import requests

url = "http://127.0.0.1:8000/predict"
file_path = r"C:\Users\anujs\Downloads\thumb.jpg"  # r"..." ensures the path is read correctly

with open(file_path, "rb") as file:
    files = {"file": file}
    response = requests.post(url, files=files)

print(response.json())  # Print response from FastAPI
