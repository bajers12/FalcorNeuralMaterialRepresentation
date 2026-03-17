from falcor import *

def render_graph_NeuralDecode():
    g = RenderGraph("NeuralDecode")

    decode = createPass("NeuralDecodePass", {
        "applyExp": True,
        "expOffset": 3.0
    })

    g.addPass(decode, "Decode")
    g.markOutput("Decode.output")
    return g

NeuralDecode = render_graph_NeuralDecode()
try:
    m.addGraph(NeuralDecode)
except NameError:
    pass