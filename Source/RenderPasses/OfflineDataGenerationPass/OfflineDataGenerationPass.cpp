/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "OfflineDataGenerationPass.h"

const char kShaderFile[] = "Renderpasses/OfflineDataGenerationPass/OfflineDataGenerationPass.cs.slang";
const int kSampleCount = 5000;
const uint32_t kThreadGroupSize = 64;

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, OfflineDataGenerationPass>();
}

struct BsdfSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
    float3 specular;
    float3 albedo;
    float3 roughness;
    float3 normal;
};

struct BsdfTestSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
};

OfflineDataGenerationPass::OfflineDataGenerationPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    //Set up program
    ProgramDesc desc;
    desc.addShaderLibrary(kShaderFile);
    desc.csEntry("main");

    mpPass = ComputePass::create(pDevice, kShaderFile);

    //Initialize structured buffer for writing sample data from GPU to CPU
    mpSampleBuffer = mpDevice->createStructuredBuffer(sizeof(BsdfTestSampleData), kSampleCount, ResourceBindFlags::UnorderedAccess, MemoryType::ReadBack);
    auto var = mpPass->getRootVar();
    var["gSampleOutputBuffer"] = mpSampleBuffer;
    var["gSampleCount"] = kSampleCount;
}

Properties OfflineDataGenerationPass::getProperties() const
{
    return {};
}

RenderPassReflection OfflineDataGenerationPass::reflect(const CompileData& compileData)
{
        RenderPassReflection r;

        r.addOutput("samples", "Buffer containing samples");

        return r;
}

void OfflineDataGenerationPass::renderUI(Gui::Widgets& widget)
{
    if (widget.button("Generate BSDF Samples"))
    {
        mbShouldGenerate = true;
    }
}

void OfflineDataGenerationPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if(!mbShouldGenerate) return;
    mbShouldGenerate = false;


    uint32_t groups = (kSampleCount + (kThreadGroupSize - 1)) / kThreadGroupSize;
    mpPass->execute(pRenderContext, groups, 1, 1);

    //map buffer address to cpu so we can read it

    const BsdfTestSampleData* pData = (const BsdfTestSampleData*)mpSampleBuffer->map();

    std::ofstream f("bsdf_samples.bin", std::ios::binary);

    // write raw buffer
    f.write(reinterpret_cast<const char*>(pData), sizeof(BsdfTestSampleData) * kSampleCount);

    f.close();

    mpSampleBuffer->unmap();


}


void OfflineDataGenerationPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    auto var = mpPass->getRootVar();

    if(mpScene) {
        mpScene->bindShaderData(var["gScene"]);
    }
}
