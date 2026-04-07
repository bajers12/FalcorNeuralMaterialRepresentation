import falcor
import numpy as np
import os
from pathlib import Path
import pandas as pd
import torch

scene_path = 'C:/Users/Philip/FalcorNeuralMaterialRepresentation/media/Arcade/Arcade.pyscene'

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


def generateData():
    # Create testbed (this initializes Falcor)
    testbed = falcor.Testbed(create_window=False)  # Headless mode
    device = testbed.device

    # Create render graph
    graph = testbed.create_render_graph("OnlineDataGeneration")
    generation_pass = graph.create_pass("OnlineDataGenerationPass", "OnlineDataGenerationPass", {"materialId": 3, "sampleCount": 10})
    graph.mark_output("OnlineDataGenerationPass.output")
    testbed.render_graph = graph;

    # Load a scene with materials
    testbed.load_scene(scene_path)


    # Execute the graph
    generation_pass.generate()
    generation_pass.setRandomSeedOffset(0)
    testbed.frame()
    np_data = generation_pass.getData()
    print(np_data)
    generation_pass.releaseData()

generateData()


