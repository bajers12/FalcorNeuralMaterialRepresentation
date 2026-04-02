add_plugin(OfflineDataGenerationPass)

target_sources(OfflineDataGenerationPass PRIVATE
    OfflineDataGenerationPass.cpp
    OfflineDataGenerationPass.h
    OfflineDataGenerationPass.cs.slang
)

target_copy_shaders(OfflineDataGenerationPass RenderPasses/OfflineDataGenerationPass)
target_source_group(OfflineDataGenerationPass "RenderPasses")
