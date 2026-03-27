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
const std::string kOutputFileName = "bsdf_samples.bin";
const std::string kDataDir = "samples";

const uint32_t kThreadGroupSize = 64;

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, OfflineDataGenerationPass>();
    ScriptBindings::registerBinding(OfflineDataGenerationPass::registerBindings);
}

void OfflineDataGenerationPass::registerBindings(pybind11::module& m)
{
    pybind11::class_<OfflineDataGenerationPass, RenderPass, ref<OfflineDataGenerationPass>> pass(m, "OfflineDataGenerationPass");
    pass.def("generate", &OfflineDataGenerationPass::generate);
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
    mpDevice = pDevice;
    mbShouldGenerate = false;
    mSampleCount = 100;
    mMaterialID = 3;

    for (const auto& [key, value] : props)
    {
        if (key == "materialId")
            mMaterialID = value;
        else if (key == "sampleCount")
            mSampleCount = value;
    }

    //For readback syncronization
    mpReadbackFence = mpDevice->createFence();

    mpGpuSampleBuffer = mpDevice->createStructuredBuffer(
        sizeof(BsdfSampleData),
        mSampleCount,
        ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal
    );

    mpReadbackBuffer = mpDevice->createStructuredBuffer(
        sizeof(BsdfSampleData),
        mSampleCount,
        ResourceBindFlags::None,
        MemoryType::ReadBack
    );

    //Initialize structured buffer for writing sample data from GPU to CPU

}

Properties OfflineDataGenerationPass::getProperties() const
{
    Properties props;
    props["materialId"] = mMaterialID;
    props["sampleCount"] = mSampleCount;
    return props;
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
        generate();
    }
}

void OfflineDataGenerationPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if(!mbShouldGenerate) return;
    mbShouldGenerate = false;
    if(!mpScene) return;

    if (mpScene->getMaterialCount() == 0)
    {
        logWarning("OfflineDataGenerationPass: Scene has no materials, cannot generate samples.");
        return;
    }


    //Setup program with defines in execute, as the slang files cannot compile if no scene is available at compile time for gScene acess
    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary(kShaderFile).csEntry("main");
    auto corformances = mpScene->getTypeConformances();
    desc.addTypeConformances(corformances);

    DefineList defines;
    defines = mpScene->getSceneDefines();


    mpPass = ComputePass::create(mpDevice, desc, defines);

    //Setup bindings
    auto var = mpPass->getRootVar();

    mpScene->bindShaderData(var["gScene"]);
    var["gSampleOutputBuffer"] = mpGpuSampleBuffer;
    var["gSampleCount"] = mSampleCount;

    //Threadsgroups and execute, threadgroups should probably be improved
    uint32_t groups = (mSampleCount + (kThreadGroupSize - 1)) / kThreadGroupSize;
    mpPass->execute(pRenderContext, mSampleCount, 1, 1);
    pRenderContext->uavBarrier(mpGpuSampleBuffer.get());


    //map buffer address to cpu so we can read it using a readback buffer
    pRenderContext->copyResource(mpReadbackBuffer.get(), mpGpuSampleBuffer.get());
    pRenderContext->submit(false);
    pRenderContext->signal(mpReadbackFence.get());
    mpReadbackFence->wait();
    const BsdfSampleData* pData = (const BsdfSampleData*)mpReadbackBuffer->map();

    std::filesystem::create_directories(kDataDir);
    std::string outputPath = kDataDir + "/" + kOutputFileName;
    std::ofstream f(outputPath, std::ios::binary);
    logInfo("Writing samples to: " + outputPath);

    // write raw buffer
    f.write(reinterpret_cast<const char*>(pData), sizeof(BsdfSampleData) * mSampleCount);

    f.close();

    mpReadbackBuffer->unmap();


}

void OfflineDataGenerationPass::generate() {
    mbShouldGenerate = true;
}


void OfflineDataGenerationPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}
