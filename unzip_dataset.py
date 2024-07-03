import zipfile

with zipfile.ZipFile('math-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('data')
