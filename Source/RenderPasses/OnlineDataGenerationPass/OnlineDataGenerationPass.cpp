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
#include "OnlineDataGenerationPass.h"

const char kShaderFile[] = "RenderPasses/OnlineDataGenerationPass/OnlineDataGenerationPass.cs.slang";

const uint32_t kThreadGroupSize = 64;

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, OnlineDataGenerationPass>();
    ScriptBindings::registerBinding(OnlineDataGenerationPass::registerBindings);
}

void OnlineDataGenerationPass::registerBindings(pybind11::module& m)
{
    pybind11::class_<OnlineDataGenerationPass, RenderPass, ref<OnlineDataGenerationPass>> pass(m, "OnlineDataGenerationPass");
    pass.def("generate", &OnlineDataGenerationPass::generate);
    pass.def("setRandomSeedOffset", &OnlineDataGenerationPass::setRandomSeedOffset);
    pass.def("setSeedState", &OnlineDataGenerationPass::setSeedState);
    pass.def("setUvGrid", &OnlineDataGenerationPass::setUvGrid);
    pass.def("clearUvGrid", &OnlineDataGenerationPass::clearUvGrid);
    pass.def("getData", &OnlineDataGenerationPass::getData);
    pass.def("releaseData", &OnlineDataGenerationPass::releaseData);
}


OnlineDataGenerationPass::OnlineDataGenerationPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    mpDevice = pDevice;
    mbShouldGenerate = false;
    mIsMapped = false;
    mRunSeed = 0;
    mSeedDomain = 0;
    mGenerationIndex = 0;
    mMaterialId = 0;
    mSampleCount = 0;
    mUvGridFullWidth = 0;
    mUvGridFullHeight = 0;
    mpMappedData = nullptr;

    parseProperties(props);

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

void OnlineDataGenerationPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == "materialId") mMaterialId = value;
        else if (key == "sampleCount") mSampleCount = value;
    }
}

Properties OnlineDataGenerationPass::getProperties() const
{
    Properties props;
    props["materialId"] = mMaterialId;
    props["sampleCount"] = mSampleCount;

    return props;
}

RenderPassReflection OnlineDataGenerationPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput("output", "Dummy output");
    return r;
}

void OnlineDataGenerationPass::renderUI(Gui::Widgets& widget)
{
    if (widget.button("Generate BSDF Samples"))
    {
        generate();
    }
}

void OnlineDataGenerationPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if(!mbShouldGenerate) return;
    mbShouldGenerate = false;
    if(!mpScene) return;

    if (mpScene->getMaterialCount() == 0)
    {
        logWarning("OnlineDataGenerationPass: Scene has no materials, cannot generate samples.");
        return;
    }

    if (mMaterialId >= mpScene->getMaterialCount())
    {
        logWarning("OnlineDataGenerationPass: Invalid material index {}.", mMaterialId);
        return;
    }



    //Setup bindings
    auto var = mpPass->getRootVar();

    const auto& pMat = mpScene->getMaterials()[mMaterialId];
    if (auto pMtlx = dynamic_ref_cast<MaterialXGraphMaterial>(pMat))
    {
        pMtlx->bindGeneratedResources(var);
    }
    // Non-MaterialX materials are evaluated through the generic Falcor material
    // interface in the shader below. Only MaterialXGraphMaterial needs this
    // extra generated-resource binding step.

    mpScene->bindShaderData(var["gScene"]);
    var["gSampleOutputBuffer"] = mpGpuSampleBuffer;
    var["gSampleCount"] = mSampleCount;
    var["gMaterialId"] = mMaterialId;
    var["gRunSeed"] = mRunSeed;
    var["gSeedDomain"] = mSeedDomain;
    var["gGenerationIndex"] = mGenerationIndex;
    var["gUseUvGrid"] = mUseUvGrid;
    var["gUvGridFullWidth"] = mUvGridFullWidth;
    var["gUvGridFullHeight"] = mUvGridFullHeight;

    //Threadsgroups and execute, threadgroups should probably be improved
    uint32_t groups = (mSampleCount + (kThreadGroupSize - 1)) / kThreadGroupSize;
    mpPass->execute(pRenderContext, mSampleCount, 1, 1);
    pRenderContext->uavBarrier(mpGpuSampleBuffer.get());


    //map buffer address to cpu so we can read it using a readback buffer
    pRenderContext->copyResource(mpReadbackBuffer.get(), mpGpuSampleBuffer.get());
    pRenderContext->submit(false);
    pRenderContext->signal(mpReadbackFence.get());
    mpReadbackFence->wait();
    mpMappedData = (BsdfSampleData*)mpReadbackBuffer->map();
    mIsMapped = true;

}

pybind11::array OnlineDataGenerationPass::getData()
{
    if (!mIsMapped || mpMappedData == nullptr)
        throw std::runtime_error("Buffer not mapped. Call execute() first.");

    size_t count = mSampleCount;

    // Number of floats per sample:
    constexpr size_t N = sizeof(BsdfSampleData) / sizeof(float);

    return pybind11::array(
        pybind11::buffer_info(
            (void*)mpMappedData,
            sizeof(float),
            pybind11::format_descriptor<float>::format(),
            2,
            { count, N },
            {
                sizeof(BsdfSampleData),
                sizeof(float)
            }
        )
    );
}

void OnlineDataGenerationPass::releaseData()
{
    if (mIsMapped)
    {
        mpReadbackBuffer->unmap();
        mpMappedData = nullptr;
        mIsMapped = false;
    }
}

void OnlineDataGenerationPass::setRandomSeedOffset(uint32_t offset) {
    mRunSeed = offset;
    mSeedDomain = 0;
    mGenerationIndex = 0;
}

void OnlineDataGenerationPass::setSeedState(uint32_t runSeed, uint32_t seedDomain, uint32_t generationIndex)
{
    mRunSeed = runSeed;
    mSeedDomain = seedDomain;
    mGenerationIndex = generationIndex;
}

void OnlineDataGenerationPass::setUvGrid(uint32_t width, uint32_t height)
{
    mUseUvGrid = true;
    mUvGridFullWidth = width;
    mUvGridFullHeight = height;
}

void OnlineDataGenerationPass::clearUvGrid()
{
    mUseUvGrid = false;
    mUvGridFullWidth = 0;
    mUvGridFullHeight = 0;
}

void OnlineDataGenerationPass::setupProgram() {

}

void OnlineDataGenerationPass::generate() {
    mbShouldGenerate = true;
}

void OnlineDataGenerationPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    if(mpScene == nullptr) return;

    //Setup program with defines in execute, as the slang files cannot compile if no scene is available at compile time for gScene acess
    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary(kShaderFile).csEntry("main");
    auto corformances = mpScene->getTypeConformances();
    desc.addTypeConformances(corformances);

    DefineList defines;
    defines = mpScene->getSceneDefines();


    mpPass = ComputePass::create(mpDevice, desc, defines);
}
