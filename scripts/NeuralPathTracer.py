from falcor import *

def render_graph_NeuralPathTracer():
    g = RenderGraph("NeuralPathTracer")

    VBufferRT = createPass("VBufferRT", {
        'samplePattern': 'Stratified',
        'sampleCount': 16,
        'useAlphaTest': True
    })

    PathTracer = createPass("PathTracer", {
        'samplesPerPixel': 1,
        'useNEE': True,
        'useMIS': True,
        'useBSDFSampling': True,

        'useNeuralMaterial': True,
        'neuralMaterialID': 0,
        'neuralBasePath': r'C:/Users/s204795/FalcorNeuralMaterialRepresentation/media/NeuralMaterials/CastIronBlue512_2,5Samples/'
    })

    AccumulatePass = createPass("AccumulatePass", {
        'enabled': True,
        'precisionMode': 'Single'
    })

    ToneMapper = createPass("ToneMapper", {
        'autoExposure': False,
        'exposureCompensation': 0.0
    })

    g.addPass(VBufferRT, "VBufferRT")
    g.addPass(PathTracer, "PathTracer")
    g.addPass(AccumulatePass, "AccumulatePass")
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")

    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g

NeuralPathTracer = render_graph_NeuralPathTracer()
try:
    m.addGraph(NeuralPathTracer)
except NameError:
    pass