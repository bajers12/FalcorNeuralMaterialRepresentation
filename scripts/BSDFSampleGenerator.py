from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_MinimalPathTracer():
    g = RenderGraph('OfflineDataGenerationPass')
    g.create_pass('OfflineDataGenerationPass', 'OfflineDataGenerationPass', {})
    g.mark_output('OfflineDataGenerationPass.output')
    return g

MinimalPathTracer = render_graph_MinimalPathTracer()
try: m.addGraph(MinimalPathTracer)
except NameError: None
