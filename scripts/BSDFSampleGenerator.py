from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_BSDFSampleGenerator():
    g = RenderGraph('OfflineDataGenerationPass')
    g.create_pass('OfflineDataGenerationPass', 'OfflineDataGenerationPass', {})
    g.mark_output('OfflineDataGenerationPass.output')
    return g

BSDFSampleGenerator = render_graph_BSDFSampleGenerator()
try: m.addGraph(BSDFSampleGenerator)
except NameError: None
