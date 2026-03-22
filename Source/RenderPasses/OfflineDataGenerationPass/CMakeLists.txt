add_plugin(OfflineDataGenerationPass)

target_sources(OfflineDataGenerationPass PRIVATE
    OfflineDataGenerationPass.cpp
    OfflineDataGenerationPass.h
    OfflineDataGenerationPass.cs.slang
)

target_source_group(OfflineDataGenerationPass "RenderPasses")
