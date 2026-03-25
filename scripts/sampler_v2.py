from falcor import *


def render_graph_samplerv2():
    g = RenderGraph("samplerv2")

    PathTracer = createPass("PathTracer", {
        'samplesPerPixel': 1,
    })
    g.addPass(PathTracer, "PathTracer")

    VBufferRT = createPass("VBufferRT", {
        'samplePattern': 'Stratified',
        'sampleCount': 16,
        'useAlphaTest': True,
    })
    g.addPass(VBufferRT, "VBufferRT")

    AccumulatePass = createPass("AccumulatePass", {
        'enabled': True,
        'precisionMode': 'Single',
    })
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {
        'autoExposure': False,
        'exposureCompensation': 0.0,
    })
    g.addPass(ToneMapper, "ToneMapper")

    # Offline BSDF/data generation pass.
    # Keep the property dict empty for now unless/until the C++ pass exposes
    # settings such as materialName/materialID/sampleCount via Properties.
    OfflineDataGenerationPass = createPass("OfflineDataGenerationPass", {})
    g.addPass(OfflineDataGenerationPass, "OfflineDataGenerationPass")

    # Main on-screen rendering path: match the working PathTracer graph.
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    # Visible output.
    g.markOutput("ToneMapper.dst")

    # Also mark the offline pass output so the render graph keeps the pass alive
    # for data generation. You can switch outputs in Mogwai if needed.
    g.markOutput("OfflineDataGenerationPass.output")

    return g


samplerv2 = render_graph_samplerv2()
try:
    m.addGraph(samplerv2)
except NameError:
    None
