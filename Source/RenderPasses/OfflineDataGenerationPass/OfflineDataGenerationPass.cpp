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

const char kShaderFile[] = "RenderPasses/OfflineDataGenerationPass/OfflineDataGenerationPass.cs.slang";
const int kSampleCount = 100;
const uint32_t kThreadGroupSize = 64;

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, OfflineDataGenerationPass>();
}

struct BsdfSampleData
{
    float2 uv;
    float4 wo;
    float4 wi;
    float4 f;
    float4 specular;
    float4 albedo;
    float4 roughness;
    float4 normal;
};

struct BsdfTestSampleData
{
    float2 uv;
    float4 wo;
    float4 wi;
    float4 f;
};

OfflineDataGenerationPass::OfflineDataGenerationPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    mpDevice = pDevice;
    mbShouldGenerate = false;

    //For readback syncronization
    mpReadbackFence = mpDevice->createFence();

    mpGpuSampleBuffer = mpDevice->createStructuredBuffer(
        sizeof(BsdfTestSampleData),
        kSampleCount,
        ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal
    );

    mpReadbackBuffer = mpDevice->createStructuredBuffer(
        sizeof(BsdfTestSampleData),
        kSampleCount,
        ResourceBindFlags::None,
        MemoryType::ReadBack
    );

    //Initialize structured buffer for writing sample data from GPU to CPU

}

Properties OfflineDataGenerationPass::getProperties() const
{
    return {};
}

RenderPassReflection OfflineDataGenerationPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput("output", "Dummy output");
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
    if(!mpScene) return;

    //Setup program with defines in execute, as the slang files cannot compile if no scene is available at compile time for gScene acess
    ProgramDesc desc;
    DefineList defines;
    desc.addShaderLibrary(kShaderFile).csEntry("main");
    defines = mpScene->getSceneDefines();
    mpPass = ComputePass::create(mpDevice, desc, defines);

    //Setup bindings
    auto var = mpPass->getRootVar();

    mpScene->bindShaderData(var["gScene"]);
    var["gSampleOutputBuffer"] = mpGpuSampleBuffer;
    var["gSampleCount"] = kSampleCount;

    //Threadsgroups and execute, threadgroups should probably be improved
    uint32_t groups = (kSampleCount + (kThreadGroupSize - 1)) / kThreadGroupSize;
    mpPass->execute(pRenderContext, groups, 1, 1);

    //map buffer address to cpu so we can read it using a readback buffer
    pRenderContext->copyResource(mpReadbackBuffer.get(), mpGpuSampleBuffer.get());
    const BsdfTestSampleData* pData = (const BsdfTestSampleData*)mpReadbackBuffer->map();


    std::ofstream f("bsdf_samples.bin", std::ios::binary);

    // write raw buffer
    f.write(reinterpret_cast<const char*>(pData), sizeof(BsdfTestSampleData) * kSampleCount);

    f.close();

    mpReadbackBuffer->unmap();


}


void OfflineDataGenerationPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}
