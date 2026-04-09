import falcor
import numpy as np
import os
from pathlib import Path
import pandas as pd
import torch

scene_path = 'C:/Users/Philip/FalcorNeuralMaterialRepresentation/media/Arcade/Arcade.pyscene'

class DataGenerator():
    def __init__(self, materialId = 0, scene_path = 'C:/Users/Philip/FalcorNeuralMaterialRepresentation/media/Arcade/Arcade.pyscene', sampleCount = 10):
        self.testbed = falcor.Testbed(create_window=False)
        self.device = device = self.testbed.device
        self.graph = self.testbed.create_render_graph("OnlineDataGeneration")
        self.generation_pass = self.graph.create_pass("OnlineDataGenerationPass", "OnlineDataGenerationPass", {"materialId": materialId, "sampleCount": sampleCount})
        self.graph.mark_output("OnlineDataGenerationPass.output")
        self.testbed.render_graph = self.graph;

        self.testbed.load_scene(scene_path)

    def generate_data(self, randomSeed: int):
        # Execute the graph
        self.generation_pass.setRandomSeedOffset(randomSeed)
        self.generation_pass.generate()
        self.testbed.frame()
        np_data = self.generation_pass.getData()

        return np_data

    def release_data(self):
        self.generation_pass.releaseData()


def generate_data():
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

