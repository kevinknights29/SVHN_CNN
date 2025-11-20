import h5py
import numpy as np

# Open the file
hf = h5py.File('/Users/kevinknights/Code/SVHN_CNN/data/train/digitStruct.mat', 'r')

data = hf['digitStruct']
bbox = data['bbox']

# Get first bbox
first_bbox_ref = bbox[0][0]  # [0] for array, [0] again for reference
bbox_obj = hf[first_bbox_ref]

print("Looking at 'label' field:")
label_field = bbox_obj['label']
print(f"Shape: {label_field.shape}")
label_refs = label_field[()]
print(f"Number of labels: {len(label_refs)}")

for i, ref in enumerate(label_refs):
    if isinstance(ref[0], h5py.h5r.Reference):
        deref = hf[ref[0]]
        value = deref[()]
        print(f"Label {i}: {value}")
print()

print("Looking at 'top' field:")
top_field = bbox_obj['top']
top_refs = top_field[()]

for i, ref in enumerate(top_refs):
    if isinstance(ref[0], h5py.h5r.Reference):
        deref = hf[ref[0]]
        value = deref[()]
        print(f"Top {i}: {value}")
print()

print("Looking at 'left' field:")
left_field = bbox_obj['left']
left_refs = left_field[()]

for i, ref in enumerate(left_refs):
    if isinstance(ref[0], h5py.h5r.Reference):
        deref = hf[ref[0]]
        value = deref[()]
        print(f"Left {i}: {value}")
print()

print("Looking at 'height' field:")
height_field = bbox_obj['height']
height_refs = height_field[()]

for i, ref in enumerate(height_refs):
    if isinstance(ref[0], h5py.h5r.Reference):
        deref = hf[ref[0]]
        value = deref[()]
        print(f"Height {i}: {value}")
print()

print("Looking at 'width' field:")
width_field = bbox_obj['width']
width_refs = width_field[()]

for i, ref in enumerate(width_refs):
    if isinstance(ref[0], h5py.h5r.Reference):
        deref = hf[ref[0]]
        value = deref[()]
        print(f"Width {i}: {value}")

hf.close()
