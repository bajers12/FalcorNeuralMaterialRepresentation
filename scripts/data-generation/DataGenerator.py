import falcor
from pathlib import Path

SEED_DOMAIN_TRAIN = 0
SEED_DOMAIN_VALIDATION = 1
SEED_DOMAIN_BOOTSTRAP = 2

class DataGenerator():
    def __init__(self, materialId = 0, scene_path = 'media/LayeredMaterial/ThreeLayeredGGXPreview.pyscene', sampleCount = 10000):
        # Construct path relative to project root
        project_root = Path(__file__).parent.parent.parent
        full_scene_path = project_root / scene_path

        self.testbed = falcor.Testbed(create_window=False)
        self.device = device = self.testbed.device
        self.graph = self.testbed.create_render_graph("OnlineDataGeneration")
        self.generation_pass = self.graph.create_pass("OnlineDataGenerationPass", "OnlineDataGenerationPass", {"materialId": materialId, "sampleCount": sampleCount})
        self.graph.mark_output("OnlineDataGenerationPass.output")
        self.testbed.render_graph = self.graph;

        self.testbed.load_scene(str(full_scene_path))

    def supports_uv_grid(self):
        return hasattr(self.generation_pass, "setUvGrid") and hasattr(
            self.generation_pass, "clearUvGrid"
        )

    def generate_data(self, run_seed: int, seed_domain: int, generation_index: int):
        # Execute the graph
        self.generation_pass.setSeedState(run_seed, seed_domain, generation_index)
        self.generation_pass.generate()
        self.testbed.frame()
        np_data = self.generation_pass.getData()

        return np_data

    def generate_grid_data(self, width: int, height: int, run_seed: int, seed_domain: int, generation_index: int = 0):
        if not self.supports_uv_grid():
            raise RuntimeError(
                "The loaded OnlineDataGenerationPass plugin does not expose setUvGrid/clearUvGrid. "
                "Rebuild Falcor so the updated render pass bindings are available, or run without encoder bootstrap."
            )
        self.generation_pass.setUvGrid(width, height)
        try:
            self.generation_pass.setSeedState(run_seed, seed_domain, generation_index)
            self.generation_pass.generate()
            self.testbed.frame()
            np_data = self.generation_pass.getData()
        finally:
            self.generation_pass.clearUvGrid()
        return np_data

    def release_data(self):
        self.generation_pass.releaseData()
