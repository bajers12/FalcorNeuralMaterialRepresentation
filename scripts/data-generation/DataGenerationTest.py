import numpy as np
import pandas as pd

# Define the dtype for BsdfTestSampleData
dt = np.dtype([
    ('uv', 'f4', (2,)),        # float2 uv
    ('wo', 'f4', (3,)),        # float4 wo
    ('wi', 'f4', (3,)),        # float4 wi
    ('f', 'f4', (3,))          # float4 f
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
    print()

# Optional: Convert to pandas DataFrame (flattening the arrays)
# Flatten the structured array into a regular array
flat_data = np.column_stack([
    data['uv'][:, 0], data['uv'][:, 1],  # uv_x, uv_y
    data['wo'][:, 0], data['wo'][:, 1], data['wo'][:, 2],  # wo_x, wo_y, wo_z, wo_w
    data['wi'][:, 0], data['wi'][:, 1], data['wi'][:, 2],  # wi_x, wi_y, wi_z, wi_w
    data['f'][:, 0], data['f'][:, 1], data['f'][:, 2]       # f_x, f_y, f_z, f_w
])

columns = ['uv_x', 'uv_y', 'wo_x', 'wo_y', 'wo_z', 'wi_x', 'wi_y', 'wi_z', 'f_x', 'f_y', 'f_z']
df = pd.DataFrame(flat_data, columns=columns)

print("Pandas DataFrame head:")
print(df.describe())
print(df.head())
print(df.tail())
