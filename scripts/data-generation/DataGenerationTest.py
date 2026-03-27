import numpy as np
import pandas as pd

# Define the dtype for BsdfTestSampleData
dt = np.dtype([
    ('uv', 'f4', (2,)),
    ('wo', 'f4', (3,)),
    ('wi', 'f4', (3,)),
    ('f', 'f4', (3,)),
    ('specular', 'f4', (3,)),
    ('albedo', 'f4', (3,)),
    ('normal', 'f4', (3,))
    ('roughness', 'f', (1,)),
])

# Read the binary file
file_path = 'C:/Users/Philip/FalcorNeuralMaterialRepresentation/build/windows-ninja-msvc/bin/Release/samples/bsdf_samples.bin'
data = np.fromfile(file_path, dtype=dt)

print(f"Loaded {len(data)} samples")
print(f"Data shape: {data.shape}")
print(f"Data dtype: {data.dtype}")

# Print first few samples
print("\nFirst 5 samples:")
for i in range(min(5, len(data))):
    print(f"Sample {i}:")
    print(f"  uv: {data[i]['uv']}")
    print(f"  wo: {data[i]['wo']}")
    print(f"  wi: {data[i]['wi']}")
    print(f"  f: {data[i]['f']}")
    print(f"  specular: {data[i]['specular']}")
    print(f"  albedo: {data[i]['albedo']}")
    print(f"  normal: {data[i]['normal']}")
    print(f"  roughness: {data[i]['roughness']}")


# Optional: Convert to pandas DataFrame (flattening the arrays)
# Flatten the structured array into a regular array
flat_data = np.column_stack([
    data['uv'][:, 0], data['uv'][:, 1],  # uv_x, uv_y
    data['wo'][:, 0], data['wo'][:, 1], data['wo'][:, 2],
    data['wi'][:, 0], data['wi'][:, 1], data['wi'][:, 2],
    data['f'][:, 0], data['f'][:, 1], data['f'][:, 2],
    data['specular'][:, 0], data['specular'][:, 1], data['specular'][:, 2],
    data['albedo'][:, 0], data['albedo'][:, 1], data['albedo'][:, 2],
    data['normal'][:, 0], data['normal'][:, 1], data['normal'][:, 2],
    data['roughness'][:, 0],
])

columns = ['uv_x', 'uv_y', 'wo_x', 'wo_y', 'wo_z', 'wi_x', 'wi_y', 'wi_z', 'f_x', 'f_y', 'f_z',
           'spec_x', 'spec_y','spec_z', 'albedo_x', 'albedo_y', 'albedo_z',  'normal_x', 'normal_y', 'normal_z', 'roughness']
df = pd.DataFrame(flat_data, columns=columns)

print("Pandas DataFrame head:")
print(df.describe())
print(df.head())
print(df.tail())
