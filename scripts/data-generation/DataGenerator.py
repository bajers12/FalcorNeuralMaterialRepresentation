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

    def supports_uv_grid(self):
        return hasattr(self.generation_pass, "setUvGrid") and hasattr(
            self.generation_pass, "clearUvGrid"
        )

    def generate_data(self, randomSeed: int):
        if self.supports_uv_grid():
            self.generation_pass.clearUvGrid()
        # Execute the graph
        self.generation_pass.setRandomSeedOffset(randomSeed)
        self.generation_pass.generate()
        self.testbed.frame()
        np_data = self.generation_pass.getData()

        return np_data

    def generate_grid_data(self, width: int, height: int, randomSeed: int = 0):
        return self.generate_grid_region_data(width, height, width, height, 0, 0, randomSeed)

    def generate_grid_region_data(
        self,
        full_width: int,
        full_height: int,
        region_width: int,
        region_height: int,
        offset_x: int,
        offset_y: int,
        randomSeed: int = 0,
    ):
        if not self.supports_uv_grid():
            raise RuntimeError(
                "The loaded OnlineDataGenerationPass plugin does not expose setUvGrid/clearUvGrid. "
                "Rebuild Falcor so the updated render pass bindings are available, or run without encoder bootstrap."
            )
        if hasattr(self.generation_pass, "setUvGridRegion"):
            self.generation_pass.setUvGridRegion(
                full_width,
                full_height,
                region_width,
                region_height,
                offset_x,
                offset_y,
            )
        else:
            if (
                region_width != full_width
                or region_height != full_height
                or offset_x != 0
                or offset_y != 0
            ):
                raise RuntimeError(
                    "The loaded OnlineDataGenerationPass plugin supports only full-grid sampling. "
                    "Rebuild Falcor so setUvGridRegion is available for tiled bootstrap."
                )
            self.generation_pass.setUvGrid(full_width, full_height)
        self.generation_pass.setRandomSeedOffset(randomSeed)
        self.generation_pass.generate()
        self.testbed.frame()
        np_data = self.generation_pass.getData()
        self.generation_pass.clearUvGrid()
        return np_data

    def release_data(self):
        self.generation_pass.releaseData()
