
import requests

url = "https://zenodo.org/api/records/4010759"

print("Checking official INCLUDE dataset files...")

response = requests.get(url, timeout=60)
response.raise_for_status()

record = response.json()

for item in record.get("files", []):
    print(
        "Name:", item.get("key"),
        "| Size:", item.get("size"),
        "| URL:", item.get("links", {}).get("self")
    )
