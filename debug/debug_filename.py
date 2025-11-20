import h5py
from pathlib import Path
import numpy as np

hf = h5py.File('/Users/kevinknights/Code/SVHN_CNN/data/train/digitStruct.mat', 'r')
data = hf['digitStruct']
name_data = data['name']

# Test first filename
first_name_ref = name_data[0][0]
print(f"first_name_ref type: {type(first_name_ref)}")
print(f"first_name_ref is Reference?: {isinstance(first_name_ref, h5py.h5r.Reference)}")

# Dereference
obj = hf[first_name_ref]
print(f"Dereferenced obj: {obj}")
string_data = obj[()]
print(f"String data: {string_data}")
print(f"String data type: {type(string_data)}")

# Convert to string
filename = ''.join(chr(c) for c in string_data.flatten())
print(f"Filename: {filename}")

# Try to read the file
train_dir = Path("/Users/kevinknights/Code/SVHN_CNN/data/train")
img_path = train_dir / filename
print(f"Image path: {img_path}")
print(f"Image exists?: {img_path.exists()}")

hf.close()
