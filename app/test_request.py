import requests

url = 'http://localhost:5000/predict'
file_path = 'dataset/flowers/daisy/5673551_01d1ea993e_n.jpg'

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print('Status Code:', response.status_code)
print('Response JSON:', response.json())