#include "NeuralMaterialViewer.h"

namespace
{
    const char kInputDecoded[] = "decoded";
    const char kOutputPassthrough[] = "passthrough";
    const char kMaterialID[] = "materialID";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NeuralMaterialViewer>();
}

ref<NeuralMaterialViewer> NeuralMaterialViewer::create(ref<Device> pDevice, const Properties& props)
{
    return make_ref<NeuralMaterialViewer>(pDevice, props);
}

NeuralMaterialViewer::NeuralMaterialViewer(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
    , mpDevice(pDevice)
{
    if (props.has(kMaterialID))
        mMaterialID = (uint32_t)props[kMaterialID];
}

Properties NeuralMaterialViewer::getProperties() const
{
    Properties props;
    props[kMaterialID] = mMaterialID;
    return props;
}

RenderPassReflection NeuralMaterialViewer::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addInput(kInputDecoded, "Decoded neural texture");
    r.addOutput(kOutputPassthrough, "Passthrough output")
        .format(ResourceFormat::RGBA32Float);
    return r;
}
void NeuralMaterialViewer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    if (!mpScene)
    {
        logWarning("NeuralMaterialViewer: no scene set.");
        return;
    }

    const auto& materials = mpScene->getMaterials();
    logInfo("NeuralMaterialViewer: scene has {} material(s).", materials.size());

    for (size_t i = 0; i < materials.size(); ++i)
    {
        auto pMaterial = materials[i];
        if (!pMaterial)
        {
            logInfo("  material[{}] = <null>", i);
            continue;
        }

        logInfo("  material[{}] name='{}'", i, pMaterial->getName());
    }
}

void NeuralMaterialViewer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pDecoded = renderData.getTexture("decoded");
    auto pOut = renderData.getTexture("passthrough");
    if (!pDecoded || !pOut) return;

    // Keep the pass alive and keep the debug thumbnail.
    pRenderContext->copyResource(pOut.get(), pDecoded.get());

    const auto& materials = mpScene->getMaterials();
    if (mMaterialID >= materials.size())
    {
        logWarning("NeuralMaterialViewer: material index {} out of range ({} materials).", mMaterialID, materials.size());
        return;
    }

    auto pMat = materials[mMaterialID];
    if (!pMat)
    {
        logWarning("NeuralMaterialViewer: material {} is null.", mMaterialID);
        return;
    }

    // Bind decoded texture directly as base color.
    pMat->setTexture(Material::TextureSlot::BaseColor, pDecoded);

    // Optional: force roughness/metallic to something neutral later if needed.
}
void NeuralMaterialViewer::renderUI(Gui::Widgets& widget)
{
    widget.var("Material ID", mMaterialID, 0u, 1024u);
}