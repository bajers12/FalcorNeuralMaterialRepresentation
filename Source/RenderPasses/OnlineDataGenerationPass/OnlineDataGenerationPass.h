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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include <fstream>
#include <filesystem>
#include "Scene/Material/MaterialXGraphMaterial.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace Falcor;

struct BsdfSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
    float3 albedo;
    float mipLevel;
};

struct BsdfFeatureSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
    float3 albedo;
    float mipLevel;
    float4 bootstrapFeature0;
    float4 bootstrapFeature1;
    float4 bootstrapFeature2;
    float4 bootstrapFeature3;
    float4 bootstrapFeature4;
    float4 bootstrapFeature5;
    float4 bootstrapFeature6;
    float4 bootstrapFeature7;
};

class OnlineDataGenerationPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(OnlineDataGenerationPass, "OnlineDataGenerationPass", "Insert pass description here.");

    static ref<OnlineDataGenerationPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<OnlineDataGenerationPass>(pDevice, props);
    }

    OnlineDataGenerationPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    pybind11::array getData();
    void releaseData();
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    void generate();
    void OnlineDataGenerationPass::setRandomSeedOffset(uint32_t offset);
    void setSeedState(uint32_t runSeed, uint32_t seedDomain, uint32_t generationIndex);
    void setUvGrid(uint32_t width, uint32_t height);
    void clearUvGrid();
    void setUvSamples(pybind11::array uvSamples);
    void clearUvSamples();
    void setMollification(float coneAngleRadians, uint32_t sampleCount);
    void setHierarchicalFiltering(bool enabled, uint32_t mipCount);
    uint32_t getBootstrapFeatureDim() const;
    std::vector<std::string> getBootstrapFeatureNames() const;
    static void registerBindings(pybind11::module& m);

private:
    enum class BootstrapFeatureLayout
    {
        None,
        Auto,
        Legacy,
        Material,
    };

    void OnlineDataGenerationPass::parseProperties(const Properties& props);
    void resolveBootstrapFeatureLayout();
    void recreateSampleBuffers();
    static BootstrapFeatureLayout parseBootstrapFeatureLayout(const std::string& value);
    static std::string bootstrapFeatureLayoutToString(BootstrapFeatureLayout layout);

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;
    ref<Buffer> mpGpuSampleBuffer;
    ref<Buffer> mpReadbackBuffer;
    ref<Fence> mpReadbackFence;
    bool mbShouldGenerate;
    bool mIsMapped;
    bool mUseUvGrid = false;
    bool mUseUvSamples = false;
    uint32_t mRunSeed;
    uint32_t mSeedDomain;
    uint32_t mGenerationIndex;
    uint32_t mMaterialId;
    uint32_t mSampleCount;
    uint32_t mUvGridFullWidth = 0;
    uint32_t mUvGridFullHeight = 0;
    std::vector<float2> mUvSamples;
    ref<Buffer> mpUvSampleBuffer;
    float mMollificationConeAngleRad = 0.f;
    uint32_t mMollificationSampleCount = 1;
    bool mHierarchicalFilteringEnabled = false;
    uint32_t mHierarchicalMipCount = 1;
    uint32_t mFinestTextureWidth = 1;
    uint32_t mFinestTextureHeight = 1;
    float mMipExponentialRate = 0.7f;
    uint32_t mMinFilterSampleCount = 1;
    uint32_t mMaxFilterSampleCount = 64;
    float mGaussianFilterStdScale = 0.5f;
    bool mGenerateAlbedoTarget = false;
    BootstrapFeatureLayout mRequestedBootstrapFeatureLayout = BootstrapFeatureLayout::Auto;
    BootstrapFeatureLayout mActiveBootstrapFeatureLayout = BootstrapFeatureLayout::None;
    std::vector<std::string> mBootstrapFeatureNames;
    size_t mSampleStrideBytes = sizeof(BsdfSampleData);
    size_t mSampleFloatCount = sizeof(BsdfSampleData) / sizeof(float);
    void* mpMappedData;
};
