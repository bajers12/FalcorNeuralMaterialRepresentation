add_plugin(NeuralDecodePass)

target_sources(NeuralDecodePass PRIVATE
    NeuralDecodePass.cpp
    NeuralDecodePass.h
    NeuralDecode.cs.slang
)