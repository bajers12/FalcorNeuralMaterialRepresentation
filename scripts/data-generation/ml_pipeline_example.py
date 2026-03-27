import falcor
import numpy as np
import os
from pathlib import Path
import pandas as pd

output_dir = 'C:/Users/Philip/FalcorNeuralMaterialRepresentation/build/windows-ninja-msvc/bin/Release/samples'
file_path = 'bsdf_samples.bin'
scene_path = 'C:/Users/Philip/FalcorNeuralMaterialRepresentation/media/Arcade/Arcade.pyscene'


# Create testbed (this initializes Falcor)
testbed = falcor.Testbed(create_window=False)  # Headless mode
device = testbed.device

# Create render graph
graph = testbed.create_render_graph("OfflineDataGeneration")
generation_pass = graph.create_pass("OfflineDataGenerationPass", "OfflineDataGenerationPass", {"materialId": 3, "sampleCount": 10, "outputDirectory": output_dir, "outputFilename": file_path})
graph.mark_output("OfflineDataGenerationPass.output")
testbed.render_graph = graph;

# Load a scene with materials
testbed.load_scene(scene_path)


# Execute the graph
generation_pass.generate()
generation_pass.setRandomSeedOffset(0);
testbed.frame()

# Define the dtype for BsdfTestSampleData
dt = np.dtype([
    ('uv', 'f4', (2,)),

    ('wo', 'f4', (3,)),

    ('wi', 'f4', (3,)),

    ('f', 'f4', (3,)),

    ('specular', 'f4', (3,)),

    ('albedo', 'f4', (3,)),

    ('normal', 'f4', (3,)),

    ('roughness', 'f4'),
    ('pdf', 'f4'),
])

# Read the binary file
data = np.fromfile(output_dir +"/" + file_path, dtype=dt)

print(f"Loaded {len(data)} samples")
print(f"Data shape: {data.shape}")
print(f"Data dtype: {data.dtype}")
print(f"data size: {data.itemsize}")

# Print first few samples
#print("\nFirst 5 samples:")
#for i in range(min(5, len(data))):
#    print(f"Sample {i}:")
#    print(f"  uv: {data[i]['uv']}")
#    print(f"  wo: {data[i]['wo']}")
#    print(f"  wi: {data[i]['wi']}")
#    print(f"  f: {data[i]['f']}")
#    print(f"  specular: {data[i]['specular']}")
#    print(f"  albedo: {data[i]['albedo']}")
#    print(f"  normal: {data[i]['normal']}")
#    print(f"  roughness: {data[i]['roughness']}")
#    print(f"  pdf: {data[i]['pdf']}")

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
    data['roughness'][:],
    data['pdf'][:]
])

columns = ['uv_x', 'uv_y', 'wo_x', 'wo_y', 'wo_z', 'wi_x', 'wi_y', 'wi_z', 'f_x', 'f_y', 'f_z',
           'spec_x', 'spec_y','spec_z', 'albedo_x', 'albedo_y', 'albedo_z',  'normal_x', 'normal_y', 'normal_z', 'roughness', 'pdf']
df = pd.DataFrame(flat_data, columns=columns)

print("Pandas DataFrame:")
print(df.describe())
print(df.head())
print(df.tail())
