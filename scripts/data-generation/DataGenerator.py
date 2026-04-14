import falcor
import numpy as np
import os
from pathlib import Path
import pandas as pd
import random

class DataGenerator():
    def __init__(self, materialId = 0, scene_path = 'C:/Users/Philip/FalcorNeuralMaterialRepresentation/MatXScenes/Preview/MatXScene.pyscene', sampleCount = 10):
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
