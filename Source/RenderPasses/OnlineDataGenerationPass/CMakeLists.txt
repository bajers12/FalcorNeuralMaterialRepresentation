add_plugin(OnlineDataGenerationPass)

target_sources(OnlineDataGenerationPass PRIVATE
    OnlineDataGenerationPass.cpp
    OnlineDataGenerationPass.h
    OnlineDataGenerationPass.cs.slang
)

target_copy_shaders(OnlineDataGenerationPass RenderPasses/OnlineDataGenerationPass)
target_source_group(OnlineDataGenerationPass "RenderPasses")
