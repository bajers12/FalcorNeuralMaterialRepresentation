#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

class NeuralMaterialViewer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(
        NeuralMaterialViewer,
        "NeuralMaterialViewer",
        "Bind decoded neural texture to a scene material"
    );

    static ref<NeuralMaterialViewer> create(ref<Device> pDevice, const Properties& props);
    NeuralMaterialViewer(ref<Device> pDevice, const Properties& props);

    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    

private:
    ref<Device> mpDevice;
    ref<Scene> mpScene;

    uint32_t mMaterialID = 0;
    bool mLoggedSceneMaterials = false;
    ref<Texture> mpBoundTexture;
    bool mMaterialBound = false;
};