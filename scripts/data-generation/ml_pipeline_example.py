import falcor
import numpy as np
import os
from pathlib import Path

# Create testbed (this initializes Falcor)
testbed = falcor.Testbed(create_window=False)  # Headless mode
device = testbed.device

# Create render graph
graph = testbed.create_render_graph("OfflineDataGeneration")
generation_pass = graph.create_pass("OfflineDataGenerationPass", "OfflineDataGenerationPass", {"materialId": 3, "sampleCount": 200})
graph.mark_output("OfflineDataGenerationPass.output")
testbed.render_graph = graph;

# Load a scene with materials
scene_path = Path('media/Arcade/Arcade.pyscene')
testbed.load_scene("C:/Users/Philip/FalcorNeuralMaterialRepresentation/media/Arcade\Arcade.pyscene")


# Execute the graph
generation_pass.generate()
testbed.frame()
