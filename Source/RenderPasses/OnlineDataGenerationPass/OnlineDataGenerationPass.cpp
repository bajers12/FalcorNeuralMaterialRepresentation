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
#include <algorithm>

const char kShaderFile[] = "RenderPasses/OnlineDataGenerationPass/OnlineDataGenerationPass.cs.slang";
const char kBootstrapFeatureLayout[] = "bootstrapFeatureLayout";
const char kHierarchicalFilteringEnabled[] = "hierarchicalFilteringEnabled";
const char kHierarchicalMipCount[] = "hierarchicalMipCount";
const char kFinestTextureWidth[] = "finestTextureWidth";
const char kFinestTextureHeight[] = "finestTextureHeight";
const char kMipExponentialRate[] = "mipExponentialRate";
const char kMinFilterSampleCount[] = "minFilterSampleCount";
const char kMaxFilterSampleCount[] = "maxFilterSampleCount";
const char kGaussianFilterStdScale[] = "gaussianFilterStdScale";
const char kGenerateAlbedoTarget[] = "generateAlbedoTarget";
const uint32_t kBootstrapFeatureCapacity = 32;

namespace
{
    const std::vector<std::string> kLegacyFeatureNames = {
        "specular.r",
        "specular.g",
        "specular.b",
        "albedo.r",
        "albedo.g",
        "albedo.b",
        "normal.x",
        "normal.y",
        "normal.z",
        "roughness",
        "pdf",
    };
}

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
    pass.def("setUvSamples", &OnlineDataGenerationPass::setUvSamples);
    pass.def("clearUvSamples", &OnlineDataGenerationPass::clearUvSamples);
    pass.def("setMollification", &OnlineDataGenerationPass::setMollification);
    pass.def("setHierarchicalFiltering", &OnlineDataGenerationPass::setHierarchicalFiltering);
    pass.def("getBootstrapFeatureDim", &OnlineDataGenerationPass::getBootstrapFeatureDim);
    pass.def("getBootstrapFeatureNames", &OnlineDataGenerationPass::getBootstrapFeatureNames);
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
    mMollificationConeAngleRad = 0.f;
    mMollificationSampleCount = 1;
    mHierarchicalFilteringEnabled = false;
    mHierarchicalMipCount = 1;
    mFinestTextureWidth = 1;
    mFinestTextureHeight = 1;
    mMipExponentialRate = 0.7f;
    mMinFilterSampleCount = 1;
    mMaxFilterSampleCount = 64;
    mGaussianFilterStdScale = 0.5f;
    mGenerateAlbedoTarget = false;
    mpMappedData = nullptr;

    parseProperties(props);

    //For readback syncronization
    mpReadbackFence = mpDevice->createFence();

    recreateSampleBuffers();
}

void OnlineDataGenerationPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == "materialId") mMaterialId = value;
        else if (key == "sampleCount") mSampleCount = value;
        else if (key == kBootstrapFeatureLayout) mRequestedBootstrapFeatureLayout = parseBootstrapFeatureLayout(value);
        else if (key == kHierarchicalFilteringEnabled) mHierarchicalFilteringEnabled = value;
        else if (key == kHierarchicalMipCount) mHierarchicalMipCount = std::max(1u, (uint32_t)value);
        else if (key == kFinestTextureWidth) mFinestTextureWidth = std::max(1u, (uint32_t)value);
        else if (key == kFinestTextureHeight) mFinestTextureHeight = std::max(1u, (uint32_t)value);
        else if (key == kMipExponentialRate) mMipExponentialRate = std::max(1e-6f, (float)value);
        else if (key == kMinFilterSampleCount) mMinFilterSampleCount = std::max(1u, (uint32_t)value);
        else if (key == kMaxFilterSampleCount) mMaxFilterSampleCount = std::max(1u, (uint32_t)value);
        else if (key == kGaussianFilterStdScale) mGaussianFilterStdScale = std::max(0.f, (float)value);
        else if (key == kGenerateAlbedoTarget) mGenerateAlbedoTarget = value;
    }
    mMaxFilterSampleCount = std::max(mMinFilterSampleCount, mMaxFilterSampleCount);
}

Properties OnlineDataGenerationPass::getProperties() const
{
    Properties props;
    props["materialId"] = mMaterialId;
    props["sampleCount"] = mSampleCount;
    props[kBootstrapFeatureLayout] = bootstrapFeatureLayoutToString(mRequestedBootstrapFeatureLayout);
    props[kHierarchicalFilteringEnabled] = mHierarchicalFilteringEnabled;
    props[kHierarchicalMipCount] = mHierarchicalMipCount;
    props[kFinestTextureWidth] = mFinestTextureWidth;
    props[kFinestTextureHeight] = mFinestTextureHeight;
    props[kMipExponentialRate] = mMipExponentialRate;
    props[kMinFilterSampleCount] = mMinFilterSampleCount;
    props[kMaxFilterSampleCount] = mMaxFilterSampleCount;
    props[kGaussianFilterStdScale] = mGaussianFilterStdScale;
    props[kGenerateAlbedoTarget] = mGenerateAlbedoTarget;

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
    var["gUseUvSamples"] = mUseUvSamples;
    var["gUvGridFullWidth"] = mUvGridFullWidth;
    var["gUvGridFullHeight"] = mUvGridFullHeight;
    if (mUseUvSamples) var["gUvSampleBuffer"] = mpUvSampleBuffer;
    var["gMollificationConeAngleRad"] = mMollificationConeAngleRad;
    var["gMollificationSampleCount"] = mMollificationSampleCount;
    var["gHierarchicalFilteringEnabled"] = mHierarchicalFilteringEnabled;
    var["gHierarchicalMipCount"] = mHierarchicalMipCount;
    var["gFinestTextureSize"] = uint2(mFinestTextureWidth, mFinestTextureHeight);
    var["gMipExponentialRate"] = mMipExponentialRate;
    var["gMinFilterSampleCount"] = mMinFilterSampleCount;
    var["gMaxFilterSampleCount"] = mMaxFilterSampleCount;
    var["gGaussianFilterStdScale"] = mGaussianFilterStdScale;
    var["gGenerateAlbedoTarget"] = mGenerateAlbedoTarget;

    mpPass->execute(pRenderContext, mSampleCount, 1, 1);
    pRenderContext->uavBarrier(mpGpuSampleBuffer.get());


    //map buffer address to cpu so we can read it using a readback buffer
    pRenderContext->copyResource(mpReadbackBuffer.get(), mpGpuSampleBuffer.get());
    pRenderContext->submit(false);
    pRenderContext->signal(mpReadbackFence.get());
    mpReadbackFence->wait();
    mpMappedData = mpReadbackBuffer->map();
    mIsMapped = true;

}

pybind11::array OnlineDataGenerationPass::getData()
{
    if (!mIsMapped || mpMappedData == nullptr)
        throw std::runtime_error("Buffer not mapped. Call execute() first.");

    size_t count = mSampleCount;

    return pybind11::array(
        pybind11::buffer_info(
            (void*)mpMappedData,
            sizeof(float),
            pybind11::format_descriptor<float>::format(),
            2,
            { count, mSampleFloatCount },
            {
                mSampleStrideBytes,
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

void OnlineDataGenerationPass::setUvSamples(pybind11::array uvSamples)
{
    pybind11::buffer_info info = uvSamples.request();
    if (info.ndim != 2 || info.shape[1] != 2)
    {
        FALCOR_THROW("OnlineDataGenerationPass::setUvSamples expects a float array with shape [N, 2].");
    }

    const uint32_t sampleCount = (uint32_t)info.shape[0];
    if (sampleCount != mSampleCount)
    {
        FALCOR_THROW(
            "OnlineDataGenerationPass::setUvSamples received {} UVs, but this pass was created with sampleCount={}.",
            sampleCount,
            mSampleCount
        );
    }

    mUvSamples.resize(sampleCount);
    const float* data = static_cast<const float*>(info.ptr);
    const size_t stride0 = (size_t)info.strides[0] / sizeof(float);
    const size_t stride1 = (size_t)info.strides[1] / sizeof(float);
    for (uint32_t i = 0; i < sampleCount; ++i)
    {
        mUvSamples[i] = float2(data[i * stride0 + 0 * stride1], data[i * stride0 + 1 * stride1]);
    }

    if (!mpUvSampleBuffer || mpUvSampleBuffer->getElementCount() != sampleCount)
    {
        mpUvSampleBuffer = mpDevice->createStructuredBuffer(
            sizeof(float2),
            sampleCount,
            ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            mUvSamples.data(),
            false
        );
    }
    else
    {
        mpUvSampleBuffer->setBlob(mUvSamples.data(), 0, mUvSamples.size() * sizeof(float2));
    }
    mUseUvSamples = true;
}

void OnlineDataGenerationPass::clearUvSamples()
{
    mUseUvSamples = false;
    mUvSamples.clear();
}

void OnlineDataGenerationPass::setMollification(float coneAngleRadians, uint32_t sampleCount)
{
    mMollificationConeAngleRad = std::max(0.f, coneAngleRadians);
    mMollificationSampleCount = std::max(1u, sampleCount);
}

void OnlineDataGenerationPass::setHierarchicalFiltering(bool enabled, uint32_t mipCount)
{
    mHierarchicalFilteringEnabled = enabled;
    mHierarchicalMipCount = std::max(1u, mipCount);
}

uint32_t OnlineDataGenerationPass::getBootstrapFeatureDim() const
{
    return (uint32_t)mBootstrapFeatureNames.size();
}

std::vector<std::string> OnlineDataGenerationPass::getBootstrapFeatureNames() const
{
    return mBootstrapFeatureNames;
}

void OnlineDataGenerationPass::generate() {
    mbShouldGenerate = true;
}

OnlineDataGenerationPass::BootstrapFeatureLayout OnlineDataGenerationPass::parseBootstrapFeatureLayout(const std::string& value)
{
    if (value == "none" || value == "off" || value == "disabled") return BootstrapFeatureLayout::None;
    if (value == "auto") return BootstrapFeatureLayout::Auto;
    if (value == "legacy") return BootstrapFeatureLayout::Legacy;
    if (value == "material" || value == "features" || value == "three_layered_ggx" || value == "ThreeLayeredGGXMaterial")
        return BootstrapFeatureLayout::Material;

    logWarning("OnlineDataGenerationPass: Unknown bootstrap feature layout '{}'. Falling back to auto.", value);
    return BootstrapFeatureLayout::Auto;
}

std::string OnlineDataGenerationPass::bootstrapFeatureLayoutToString(BootstrapFeatureLayout layout)
{
    switch (layout)
    {
    case BootstrapFeatureLayout::None:
        return "none";
    case BootstrapFeatureLayout::Auto:
        return "auto";
    case BootstrapFeatureLayout::Legacy:
        return "legacy";
    case BootstrapFeatureLayout::Material:
        return "material";
    default:
        return "legacy";
    }
}

void OnlineDataGenerationPass::resolveBootstrapFeatureLayout()
{
    mBootstrapFeatureNames.clear();
    mActiveBootstrapFeatureLayout = BootstrapFeatureLayout::None;

    if (
        mRequestedBootstrapFeatureLayout == BootstrapFeatureLayout::None ||
        mpScene == nullptr ||
        mpScene->getMaterialCount() == 0 ||
        mMaterialId >= mpScene->getMaterialCount()
    )
        return;

    if (mRequestedBootstrapFeatureLayout == BootstrapFeatureLayout::Legacy)
    {
        mActiveBootstrapFeatureLayout = BootstrapFeatureLayout::Legacy;
        mBootstrapFeatureNames = kLegacyFeatureNames;
        return;
    }

    const auto& pMat = mpScene->getMaterials()[mMaterialId];
    if (!pMat) return;

    mBootstrapFeatureNames = pMat->getBootstrapFeatureNames();
    if (mBootstrapFeatureNames.empty() && mRequestedBootstrapFeatureLayout == BootstrapFeatureLayout::Auto)
    {
        mActiveBootstrapFeatureLayout = BootstrapFeatureLayout::Legacy;
        mBootstrapFeatureNames = kLegacyFeatureNames;
        return;
    }

    if (mBootstrapFeatureNames.size() > kBootstrapFeatureCapacity)
    {
        FALCOR_THROW(
            "OnlineDataGenerationPass: Material '{}' exposes {} bootstrap features, but the sample payload only has room for {}.",
            pMat->getName(),
            mBootstrapFeatureNames.size(),
            kBootstrapFeatureCapacity
        );
    }

    if (mRequestedBootstrapFeatureLayout == BootstrapFeatureLayout::Material && mBootstrapFeatureNames.empty())
    {
        logWarning("OnlineDataGenerationPass: Material '{}' does not expose bootstrap features.", pMat->getName());
        return;
    }

    if (!mBootstrapFeatureNames.empty()) mActiveBootstrapFeatureLayout = BootstrapFeatureLayout::Material;
}

void OnlineDataGenerationPass::recreateSampleBuffers()
{
    if (mIsMapped) releaseData();

    mSampleStrideBytes = mBootstrapFeatureNames.empty() ? sizeof(BsdfSampleData) : sizeof(BsdfFeatureSampleData);
    mSampleFloatCount = mSampleStrideBytes / sizeof(float);

    mpGpuSampleBuffer = mpDevice->createStructuredBuffer(
        mSampleStrideBytes,
        mSampleCount,
        ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal
    );

    mpReadbackBuffer = mpDevice->createStructuredBuffer(
        mSampleStrideBytes,
        mSampleCount,
        ResourceBindFlags::None,
        MemoryType::ReadBack
    );
}

void OnlineDataGenerationPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    if(mpScene == nullptr) return;

    resolveBootstrapFeatureLayout();
    recreateSampleBuffers();

    //Setup program with defines in execute, as the slang files cannot compile if no scene is available at compile time for gScene acess
    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary(kShaderFile).csEntry("main");
    auto corformances = mpScene->getTypeConformances();
    desc.addTypeConformances(corformances);

    DefineList defines;
    defines = mpScene->getSceneDefines();
    defines.add("SAMPLE_FEATURES_ENABLED", mBootstrapFeatureNames.empty() ? "0" : "1");
    defines.add("LEGACY_FEATURES_ENABLED", mActiveBootstrapFeatureLayout == BootstrapFeatureLayout::Legacy ? "1" : "0");
    defines.add("MATERIAL_FEATURES_ENABLED", mActiveBootstrapFeatureLayout == BootstrapFeatureLayout::Material ? "1" : "0");


    mpPass = ComputePass::create(mpDevice, desc, defines);
}
