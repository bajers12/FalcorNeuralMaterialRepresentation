import falcor
from pathlib import Path

SEED_DOMAIN_TRAIN = 0
SEED_DOMAIN_VALIDATION = 1
SEED_DOMAIN_BOOTSTRAP = 2

class DataGenerator():
    def __init__(
        self,
        materialId = 0,
        scene_path = 'media/LayeredMaterial/ThreeLayeredGGXPreview.pyscene',
        sampleCount = 10000,
        bootstrap_feature_layout = "auto",
        hierarchical_filtering_enabled = False,
        hierarchical_mip_count = 1,
        finest_texture_width = 1,
        finest_texture_height = 1,
        mip_exponential_rate = 0.7,
        min_filter_sample_count = 1,
        max_filter_sample_count = 8,
        gaussian_filter_std_scale = 0.5,
        generate_albedo_target = False,
    ):
        # Construct path relative to project root
        project_root = Path(__file__).parent.parent.parent
        full_scene_path = project_root / scene_path

        self.testbed = falcor.Testbed(create_window=False)
        self.device = device = self.testbed.device
        self.graph = self.testbed.create_render_graph("OnlineDataGeneration")
        self.generation_pass = self.graph.create_pass(
            "OnlineDataGenerationPass",
            "OnlineDataGenerationPass",
            {
                "materialId": materialId,
                "sampleCount": sampleCount,
                "bootstrapFeatureLayout": bootstrap_feature_layout,
                "hierarchicalFilteringEnabled": hierarchical_filtering_enabled,
                "hierarchicalMipCount": hierarchical_mip_count,
                "finestTextureWidth": finest_texture_width,
                "finestTextureHeight": finest_texture_height,
                "mipExponentialRate": mip_exponential_rate,
                "minFilterSampleCount": min_filter_sample_count,
                "maxFilterSampleCount": max_filter_sample_count,
                "gaussianFilterStdScale": gaussian_filter_std_scale,
                "generateAlbedoTarget": generate_albedo_target,
            },
        )
        self.graph.mark_output("OnlineDataGenerationPass.output")
        self.testbed.render_graph = self.graph;

        self.testbed.load_scene(str(full_scene_path))

    def supports_uv_grid(self):
        return hasattr(self.generation_pass, "setUvGrid") and hasattr(
            self.generation_pass, "clearUvGrid"
        )

    def supports_uv_samples(self):
        return hasattr(self.generation_pass, "setUvSamples") and hasattr(
            self.generation_pass, "clearUvSamples"
        )

    def supports_mollification(self):
        return hasattr(self.generation_pass, "setMollification")

    def get_bootstrap_feature_names(self):
        if not hasattr(self.generation_pass, "getBootstrapFeatureNames"):
            return []
        return list(self.generation_pass.getBootstrapFeatureNames())

    def get_bootstrap_feature_dim(self):
        if not hasattr(self.generation_pass, "getBootstrapFeatureDim"):
            return 0
        return int(self.generation_pass.getBootstrapFeatureDim())

    def _set_mollification(self, cone_angle_rad: float, sample_count: int):
        active = cone_angle_rad > 0.0 and sample_count > 1
        if active and not self.supports_mollification():
            raise RuntimeError(
                "The loaded OnlineDataGenerationPass plugin does not expose setMollification. "
                "Rebuild Falcor so the updated render pass bindings are available, or disable mollification."
            )
        if self.supports_mollification():
            self.generation_pass.setMollification(float(cone_angle_rad), int(sample_count))

    def generate_data(
        self,
        run_seed: int,
        seed_domain: int,
        generation_index: int,
        mollification_cone_angle_rad: float = 0.0,
        mollification_sample_count: int = 1,
    ):
        # Execute the graph
        self._set_mollification(mollification_cone_angle_rad, mollification_sample_count)
        self.generation_pass.setSeedState(run_seed, seed_domain, generation_index)
        self.generation_pass.generate()
        self.testbed.frame()
        np_data = self.generation_pass.getData()

        return np_data

    def generate_grid_data(
        self,
        width: int,
        height: int,
        run_seed: int,
        seed_domain: int,
        generation_index: int = 0,
        mollification_cone_angle_rad: float = 0.0,
        mollification_sample_count: int = 1,
    ):
        if not self.supports_uv_grid():
            raise RuntimeError(
                "The loaded OnlineDataGenerationPass plugin does not expose setUvGrid/clearUvGrid. "
                "Rebuild Falcor so the updated render pass bindings are available, or run without encoder bootstrap."
            )
        self.generation_pass.setUvGrid(width, height)
        try:
            self._set_mollification(mollification_cone_angle_rad, mollification_sample_count)
            self.generation_pass.setSeedState(run_seed, seed_domain, generation_index)
            self.generation_pass.generate()
            self.testbed.frame()
            np_data = self.generation_pass.getData()
        finally:
            self.generation_pass.clearUvGrid()
        return np_data

    def generate_uv_data(
        self,
        uv_samples,
        run_seed: int,
        seed_domain: int,
        generation_index: int = 0,
        mollification_cone_angle_rad: float = 0.0,
        mollification_sample_count: int = 1,
    ):
        if not self.supports_uv_samples():
            raise RuntimeError(
                "The loaded OnlineDataGenerationPass plugin does not expose setUvSamples/clearUvSamples. "
                "Rebuild Falcor so the updated render pass bindings are available."
            )

        import numpy as np

        uv_samples = np.ascontiguousarray(uv_samples, dtype=np.float32)
        self.generation_pass.setUvSamples(uv_samples)
        try:
            self._set_mollification(mollification_cone_angle_rad, mollification_sample_count)
            self.generation_pass.setSeedState(run_seed, seed_domain, generation_index)
            self.generation_pass.generate()
            self.testbed.frame()
            np_data = self.generation_pass.getData()
        finally:
            self.generation_pass.clearUvSamples()
        return np_data

    def release_data(self):
        self.generation_pass.releaseData()
