import os
from pathlib import Path

from falcor import *


output_dir = Path(os.environ["NEURAL_CAPTURE_OUTPUT"]).resolve()
frame_count = int(os.environ.get("NEURAL_CAPTURE_FRAMES", "8192"))
width = int(os.environ.get("NEURAL_CAPTURE_WIDTH", "1920"))
height = int(os.environ.get("NEURAL_CAPTURE_HEIGHT", "1080"))

reference_scene = os.environ.get("REFERENCE_SCENE_PATH")

if reference_scene:
    asset_dirs = [None]
elif "NEURAL_ASSET_PATHS" in os.environ:
    asset_dirs = [
        Path(path).resolve()
        for path in os.environ["NEURAL_ASSET_PATHS"].split(os.pathsep)
        if path
    ]
elif "NEURAL_ASSET_ROOT" in os.environ:
    asset_dirs = sorted(
        path.resolve()
        for path in Path(os.environ["NEURAL_ASSET_ROOT"]).iterdir()
        if path.is_dir() and path.name.endswith("_runtime")
    )
else:
    asset_dirs = [Path(os.environ["NEURAL_ASSET_PATH"]).resolve()]

g = RenderGraph("PathTracer")

vbuffer = createPass(
    "VBufferRT",
    {
        "samplePattern": "Stratified",
        "sampleCount": 16,
        "useAlphaTest": True,
    },
)
path_tracer = createPass(
    "PathTracer",
    {
        "samplesPerPixel": 1,
    },
)
accumulate = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
tone_mapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})

g.addPass(vbuffer, "VBufferRT")
g.addPass(path_tracer, "PathTracer")
g.addPass(accumulate, "AccumulatePass")
g.addPass(tone_mapper, "ToneMapper")
g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
g.addEdge("PathTracer.color", "AccumulatePass.input")
g.addEdge("AccumulatePass.output", "ToneMapper.src")
g.markOutput("ToneMapper.dst")

m.addGraph(g)
m.resizeFrameBuffer(width, height)
m.ui = False
m.clock.pause()
m.frameCapture.outputDir = str(output_dir)

scene_path = (
    Path(reference_scene).resolve()
    if reference_scene
    else Path(__file__).resolve().parents[1]
    / "MatXScenes"
    / "Preview"
    / "NeuralSphere_Mosaic_Batch.pyscene"
)

for asset_dir in asset_dirs:
    if asset_dir is None:
        capture_name = os.environ.get("REFERENCE_CAPTURE_NAME", "reference")
    else:
        os.environ["NEURAL_ASSET_PATH"] = str(asset_dir)
        capture_name = asset_dir.name.removesuffix("_runtime")
    print(f"[batch-render] Loading {capture_name}")
    m.loadScene(str(scene_path))
    m.frameCapture.baseFilename = capture_name

    for frame in range(1, frame_count + 1):
        m.clock.frame = frame
        m.renderFrame()
        if frame % 1024 == 0:
            print(f"[batch-render] {capture_name}: {frame}/{frame_count} frames")

    m.frameCapture.capture()
    print(f"[batch-render] Captured {capture_name}")

exit()
