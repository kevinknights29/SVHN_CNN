import h5py
import numpy as np

# Open the file
hf = h5py.File('/Users/kevinknights/Code/SVHN_CNN/data/train/digitStruct.mat', 'r')

print("Top level keys:", list(hf.keys()))
print()

# Look at digitStruct
data = hf['digitStruct']
print("digitStruct keys:", list(data.keys()))
print("digitStruct shape:", data.shape if hasattr(data, 'shape') else "no shape")
print()

# Look at bbox
bbox = data['bbox']
print("bbox type:", type(bbox))
print("bbox shape:", bbox.shape if hasattr(bbox, 'shape') else "no shape")
print("bbox dtype:", bbox.dtype if hasattr(bbox, 'dtype') else "no dtype")
print()

# Look at name
name = data['name']
print("name type:", type(name))
print("name shape:", name.shape if hasattr(name, 'shape') else "no shape")
print("name dtype:", name.dtype if hasattr(name, 'dtype') else "no dtype")
print()

# Try to access first bbox
print("Trying to access first bbox...")
try:
    first_bbox = bbox[0]
    print("first_bbox type:", type(first_bbox))
    print("first_bbox value:", first_bbox)
    print("first_bbox is Reference?:", isinstance(first_bbox, h5py.h5r.Reference))
    if hasattr(first_bbox, 'shape'):
        print("first_bbox shape:", first_bbox.shape)
    if hasattr(first_bbox, '__getitem__'):
        print("first_bbox[0]:", first_bbox[0])
except Exception as e:
    print("Error:", e)

print()

# Try to access first name
print("Trying to access first name...")
try:
    first_name = name[0]
    print("first_name type:", type(first_name))
    print("first_name value:", first_name)
    print("first_name is Reference?:", isinstance(first_name, h5py.h5r.Reference))
    if hasattr(first_name, 'shape'):
        print("first_name shape:", first_name.shape)
    if hasattr(first_name, '__getitem__'):
        print("first_name[0]:", first_name[0])
        first_name_0 = first_name[0]
        print("first_name[0] type:", type(first_name_0))
        print("first_name[0] is Reference?:", isinstance(first_name_0, h5py.h5r.Reference))

        # Try to dereference
        if isinstance(first_name_0, h5py.h5r.Reference):
            print("Dereferencing...")
            obj = hf[first_name_0]
            print("Dereferenced object:", obj)
            print("Dereferenced value:", obj[()])
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()

hf.close()
