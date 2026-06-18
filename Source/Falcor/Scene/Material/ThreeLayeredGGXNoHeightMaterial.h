#pragma once

#include "Material.h"
#include "Core/API/Sampler.h"
#include "Scene/Material/MaterialTypeRegistry.h"

#include <filesystem>
#include <utility>

namespace Falcor
{
    class ThreeLayeredGGXNoHeightMaterial : public Material
    {
        FALCOR_OBJECT(ThreeLayeredGGXNoHeightMaterial)

    public:
        static ref<ThreeLayeredGGXNoHeightMaterial> create(ref<Device> pDevice, const std::string& name = "")
        {
            return make_ref<ThreeLayeredGGXNoHeightMaterial>(std::move(pDevice), name);
        }

        static ref<ThreeLayeredGGXNoHeightMaterial> create(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        {
            return make_ref<ThreeLayeredGGXNoHeightMaterial>(std::move(pDevice), name, textureDirectory);
        }

        ThreeLayeredGGXNoHeightMaterial(ref<Device> pDevice, const std::string& name);
        ThreeLayeredGGXNoHeightMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory);

        bool renderUI(Gui::Widgets& widget) override;
        UpdateFlags update(MaterialSystem* pOwner) override;
        bool isEqual(const ref<Material>& pOther) const override;
        MaterialDataBlob getDataBlob() const override;
        ProgramDesc::ShaderModuleList getShaderModules() const override;
        TypeConformanceList getTypeConformances() const override;
        std::vector<std::string> getBootstrapFeatureNames() const override;

        void setDefaultTextureSampler(const ref<Sampler>& pSampler) override { mpSampler = pSampler; }
        ref<Sampler> getDefaultTextureSampler() const override { return mpSampler; }

        void setBaseColorTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::BaseColor, pTexture); }
        ref<Texture> getBaseColorTexture() const { return getTexture(TextureSlot::BaseColor); }

        void setNormalTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::Normal, pTexture); }
        ref<Texture> getNormalTexture() const { return getTexture(TextureSlot::Normal); }

        // layerRoughnessMap.rgb = base/mid/coat roughness.
        void setLayerRoughnessTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::Emissive, pTexture); }
        ref<Texture> getLayerRoughnessTexture() const { return getTexture(TextureSlot::Emissive); }

        void setDustCoverageTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::Index, pTexture); }
        ref<Texture> getDustCoverageTexture() const { return getTexture(TextureSlot::Index); }

        bool loadTextureSet(const std::filesystem::path& textureDirectory);

    private:
        struct Data
        {
            TextureHandle texBaseColor;
            TextureHandle texNormal;
            TextureHandle texLayerRoughness;
            TextureHandle texDustCoverage;

            float baseF0 = 0.02f;
            float midF0 = 0.2f;
            float coatF0 = 0.3f;

            float roughnessScale = 1.f;
            float roughnessBias = 0.02f;

            float dustCoverageScale = 0.65f;

            float baseNormalFlatten = 0.f;
            float midNormalFlatten = 0.2f;
            float coatNormalFlatten = 0.8f;

            uint32_t enableBaseLayer = 1;
            uint32_t enableMidLayer = 1;
            uint32_t enableCoatLayer = 0;
            uint32_t flipNormalY = 1;
        };
        static_assert(sizeof(Data) <= sizeof(MaterialPayload), "ThreeLayeredGGXNoHeightMaterial payload must fit in MaterialPayload");

        void setupTextureSlots();
        ref<Texture> loadExrTexture(const std::filesystem::path& path, bool singleChannel) const;
        static void renderTextureInfo(Gui::Widgets& widget, const char* label, const ref<Texture>& pTexture);

        Data mData = {};
        ref<Sampler> mpSampler;
    };
}
