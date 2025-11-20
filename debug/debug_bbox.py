import h5py
import numpy as np

# Open the file
hf = h5py.File('/Users/kevinknights/Code/SVHN_CNN/data/train/digitStruct.mat', 'r')

data = hf['digitStruct']
bbox = data['bbox']

# Get first bbox
first_bbox_ref = bbox[0][0]  # [0] for array, [0] again for reference
print("first_bbox_ref is Reference?:", isinstance(first_bbox_ref, h5py.h5r.Reference))

# Dereference
bbox_obj = hf[first_bbox_ref]
print("bbox_obj keys:", list(bbox_obj.keys()))
print()

# Look at each field
for key in bbox_obj.keys():
    field = bbox_obj[key]
    print(f"{key}:")
    print(f"  type: {type(field)}")
    print(f"  shape: {field.shape if hasattr(field, 'shape') else 'no shape'}")
    print(f"  value/ref: {field[()]}")

    # Try to dereference if it's a reference
    val = field[()]
    if isinstance(val, h5py.h5r.Reference):
        deref = hf[val]
        print(f"  dereferenced: {deref[()]}")
    elif hasattr(val, '__len__') and len(val) > 0:
        if isinstance(val[0], h5py.h5r.Reference):
            for i, ref in enumerate(val):
                deref = hf[ref]
                print(f"  dereferenced[{i}]: {deref[()]}")
    print()

hf.close()
