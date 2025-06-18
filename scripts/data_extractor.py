import requests

url = "https://corp.digitalcorpora.org/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/zipfiles/0000-0999/0000.zip"
response = requests.get(url)
with open("0000.zip", "wb") as f:
    f.write(response.content)
# Unzip the file
import zipfile

with zipfile.ZipFile("0000.zip", "r") as zip_ref:
    zip_ref.extractall("0000")
# Read the PDF files
import os

print(os.listdir("0000"))

# Create a folder to store the 100 files
os.mkdir("0000_small")


# Extract 100 files, and copy them in '0000_small' folder
for i in range(100):
    os.system(f"cp 0000/{os.listdir('0000')[i]} 0000_small")
