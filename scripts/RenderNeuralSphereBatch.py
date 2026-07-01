import json
import os
import time
from pathlib import Path

from falcor import *


output_dir = Path(os.environ["NEURAL_CAPTURE_OUTPUT"]).resolve()
timing_path = output_dir / "render_timings.json"
frame_count = int(os.environ.get("NEURAL_CAPTURE_FRAMES", "8192"))
warmup_frames = max(0, int(os.environ.get("NEURAL_CAPTURE_WARMUP_FRAMES", "0")))
reset_between_checkpoints = os.environ.get(
    "NEURAL_CAPTURE_RESET_BETWEEN_CHECKPOINTS", "0"
).lower() in ("1", "true", "yes", "on")
checkpoint_text = os.environ.get("NEURAL_CAPTURE_FRAME_CHECKPOINTS", "")
if checkpoint_text.strip():
    frame_checkpoints = sorted(
        {
            int(value)
            for value in checkpoint_text.replace(",", " ").split()
            if value.strip()
        }
    )
else:
    frame_checkpoints = [frame_count]
frame_checkpoints = [frame for frame in frame_checkpoints if 0 < frame <= frame_count]
if not frame_checkpoints or frame_checkpoints[-1] != frame_count:
    frame_checkpoints.append(frame_count)
width = int(os.environ.get("NEURAL_CAPTURE_WIDTH", "1920"))
height = int(os.environ.get("NEURAL_CAPTURE_HEIGHT", "1080"))
use_bsdf_sampling = os.environ.get("NEURAL_CAPTURE_USE_BSDF_SAMPLING", "0").lower() in ("1", "true", "yes", "on")
primary_lod_mode = os.environ.get("NEURAL_PRIMARY_LOD_MODE", "RayDiffs")
capture_name_override = os.environ.get("NEURAL_CAPTURE_NAME", "")
capture_mode_suffix = os.environ.get("NEURAL_CAPTURE_MODE_SUFFIX", "")
camera_pos_x = os.environ.get("NEURAL_CAMERA_POS_X")
camera_pos_y = os.environ.get("NEURAL_CAMERA_POS_Y")
camera_pos_z = os.environ.get("NEURAL_CAMERA_POS_Z")
camera_target_x = os.environ.get("NEURAL_CAMERA_TARGET_X")
camera_target_y = os.environ.get("NEURAL_CAMERA_TARGET_Y")
camera_target_z = os.environ.get("NEURAL_CAMERA_TARGET_Z")

reference_scene = os.environ.get("REFERENCE_SCENE_PATH")

if reference_scene:
    asset_dirs = [None]
elif "NEURAL_ASSET_PATHS" in os.environ:
    asset_dirs = [
        Path(path)
        for path in os.environ["NEURAL_ASSET_PATHS"].split(os.pathsep)
        if path
    ]
elif "NEURAL_ASSET_ROOT" in os.environ:
    asset_dirs = sorted(
        path
        for path in Path(os.environ["NEURAL_ASSET_ROOT"]).iterdir()
        if path.is_dir() and path.name.endswith("_runtime")
    )
else:
    asset_dirs = [Path(os.environ["NEURAL_ASSET_PATH"])]

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
        "useBSDFSampling": use_bsdf_sampling,
        "primaryLodMode": primary_lod_mode,
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
    / "NeuralSphere_Mosaic.pyscene"
)

try:
    render_timings = json.loads(timing_path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    render_timings = {}

for asset_dir in asset_dirs:
    if asset_dir is None:
        capture_name = os.environ.get("REFERENCE_CAPTURE_NAME", "reference")
    else:
        os.environ["NEURAL_ASSET_PATH"] = str(asset_dir)
        capture_name = asset_dir.name.removesuffix("_runtime")
        if capture_mode_suffix:
            capture_name = f"{capture_name}__{capture_mode_suffix}"
    if capture_name_override:
        capture_name = capture_name_override
    print(f"[batch-render] Loading {capture_name}")
    load_start = time.perf_counter()
    m.loadScene(str(scene_path))
    if camera_pos_x is not None or camera_pos_y is not None or camera_pos_z is not None:
        camera = m.scene.camera
        pos = camera.position
        camera.position = float3(
            float(camera_pos_x) if camera_pos_x is not None else pos.x,
            float(camera_pos_y) if camera_pos_y is not None else pos.y,
            float(camera_pos_z) if camera_pos_z is not None else pos.z,
        )
        target = camera.target
        camera.target = float3(
            float(camera_target_x) if camera_target_x is not None else target.x,
            float(camera_target_y) if camera_target_y is not None else target.y,
            float(camera_target_z) if camera_target_z is not None else target.z,
        )
    scene_load_seconds = time.perf_counter() - load_start
    m.frameCapture.baseFilename = capture_name

    checkpoint_set = set(frame_checkpoints)
    capture_seconds_total = 0.0

    def render_warmup() -> float:
        if warmup_frames <= 0:
            return 0.0
        print(f"[batch-render] {capture_name}: warming up {warmup_frames} frames")
        warmup_start = time.perf_counter()
        for warmup_frame in range(1, warmup_frames + 1):
            m.clock.frame = warmup_frame
            m.renderFrame()
        return time.perf_counter() - warmup_start

    def reset_accumulation() -> None:
        accumulate.reset()
        m.clock.frame = 0

    def capture_checkpoint(frame: int, render_seconds: float, warmup_seconds: float) -> None:
        global capture_seconds_total
        capture_start = time.perf_counter()
        m.frameCapture.capture()
        capture_seconds = time.perf_counter() - capture_start
        capture_seconds_total += capture_seconds
        timing_key = (
            capture_name
            if len(frame_checkpoints) == 1
            else f"{capture_name}@{frame}"
        )
        render_timings[timing_key] = {
            "frames": frame,
            "render_seconds": render_seconds,
            "milliseconds_per_frame": 1000.0 * render_seconds / frame,
            "frames_per_second": frame / render_seconds,
            "scene_load_seconds": scene_load_seconds,
            "warmup_frames": warmup_frames,
            "warmup_seconds": warmup_seconds,
            "capture_seconds": capture_seconds,
            "use_bsdf_sampling": use_bsdf_sampling,
            "reset_between_checkpoints": reset_between_checkpoints,
        }
        timing_path.write_text(json.dumps(render_timings, indent=2), encoding="utf-8")
        print(
            f"[batch-render-timing] {capture_name}@{frame}: "
            f"render={render_seconds:.6f}s, "
            f"frame={1000.0 * render_seconds / frame:.6f}ms, "
            f"fps={frame / render_seconds:.3f}, "
            f"load={scene_load_seconds:.6f}s, warmup={warmup_seconds:.6f}s, "
            f"capture={capture_seconds:.6f}s, useBSDFSampling={use_bsdf_sampling}"
        )
        print(f"[batch-render] Captured {capture_name} at {frame} frames")

    if reset_between_checkpoints:
        for checkpoint in frame_checkpoints:
            warmup_seconds = render_warmup()
            reset_accumulation()
            render_start = time.perf_counter()
            for frame in range(1, checkpoint + 1):
                m.clock.frame = frame
                m.renderFrame()
            render_seconds = time.perf_counter() - render_start
            capture_checkpoint(checkpoint, render_seconds, warmup_seconds)
    else:
        warmup_seconds = render_warmup()
        reset_accumulation()
        render_start = time.perf_counter()
        for frame in range(1, frame_count + 1):
            m.clock.frame = frame
            m.renderFrame()
            if frame % 1024 == 0:
                print(f"[batch-render] {capture_name}: {frame}/{frame_count} frames")
            if frame in checkpoint_set:
                render_seconds = time.perf_counter() - render_start
                capture_checkpoint(frame, render_seconds, warmup_seconds)

    print(
        f"[batch-render] Finished {capture_name}; "
        f"captured checkpoints={frame_checkpoints}, "
        f"capture_time_total={capture_seconds_total:.6f}s"
    )

exit()
