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

using namespace Falcor;

struct BsdfSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
    float3 specular;
    float3 albedo;
    float3 normal;
    float1 roughness;
    float1 pdf;
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
    void setUvGrid(uint32_t width, uint32_t height);
    void clearUvGrid();
    static void registerBindings(pybind11::module& m);

private:
    void OnlineDataGenerationPass::parseProperties(const Properties& props);
    void OnlineDataGenerationPass::setupProgram();

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;
    ref<Buffer> mpGpuSampleBuffer;
    ref<Buffer> mpReadbackBuffer;
    ref<Fence> mpReadbackFence;
    bool mbShouldGenerate;
    bool mIsMapped;
    bool mUseUvGrid = false;
    uint32_t mRandomSeedOffset;
    uint32_t mMaterialId;
    uint32_t mSampleCount;
    uint32_t mUvGridFullWidth = 0;
    uint32_t mUvGridFullHeight = 0;
    BsdfSampleData* mpMappedData;
};
