#pragma once

#include "Material.h"
#include "Core/API/Sampler.h"
#include "Scene/Material/MaterialTypeRegistry.h"

#include <filesystem>
#include <utility>

namespace Falcor
{
    class LayeredTextureMaterial : public Material
    {
        FALCOR_OBJECT(LayeredTextureMaterial)

    public:
        static ref<LayeredTextureMaterial> create(ref<Device> pDevice, const std::string& name = "")
        {
            return make_ref<LayeredTextureMaterial>(std::move(pDevice), name);
        }

        static ref<LayeredTextureMaterial> create(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        {
            return make_ref<LayeredTextureMaterial>(std::move(pDevice), name, textureDirectory);
        }

        LayeredTextureMaterial(ref<Device> pDevice, const std::string& name);
        LayeredTextureMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory);

        bool renderUI(Gui::Widgets& widget) override;
        UpdateFlags update(MaterialSystem* pOwner) override;
        bool isEqual(const ref<Material>& pOther) const override;
        MaterialDataBlob getDataBlob() const override;
        ProgramDesc::ShaderModuleList getShaderModules() const override;
        TypeConformanceList getTypeConformances() const override;

        void setDefaultTextureSampler(const ref<Sampler>& pSampler) override { mpSampler = pSampler; }
        ref<Sampler> getDefaultTextureSampler() const override { return mpSampler; }

        void setBaseColorTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::BaseColor, pTexture); }
        ref<Texture> getBaseColorTexture() const { return getTexture(TextureSlot::BaseColor); }

        void setRoughnessTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::Specular, pTexture); }
        ref<Texture> getRoughnessTexture() const { return getTexture(TextureSlot::Specular); }

        void setNormalTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::Normal, pTexture); }
        ref<Texture> getNormalTexture() const { return getTexture(TextureSlot::Normal); }

        void setDisplacementTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::Displacement, pTexture); }
        ref<Texture> getDisplacementTexture() const { return getTexture(TextureSlot::Displacement); }

        // Packed maps:
        //  - layerRoughnessMap.rgb = base/mid/coat roughness overrides
        //  - layerWeightMap.rgb    = base/mid/coat layer weights
        void setLayerRoughnessTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::Emissive, pTexture); }
        ref<Texture> getLayerRoughnessTexture() const { return getTexture(TextureSlot::Emissive); }

        void setLayerWeightTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::Transmission, pTexture); }
        ref<Texture> getLayerWeightTexture() const { return getTexture(TextureSlot::Transmission); }

        bool loadTextureSet(const std::filesystem::path& textureDirectory);

    private:
        struct Data
        {
            TextureHandle texBaseColor;
            TextureHandle texRoughness;
            TextureHandle texNormal;
            TextureHandle texDisplacement;
            TextureHandle texLayerRoughness;
            TextureHandle texLayerWeights;

            float baseF0 = 0.04f;
            float midF0 = 0.06f;
            float coatF0 = 0.08f;

            float roughnessScale = 1.f;
            float roughnessBias = 0.f;
            float thicknessScale = 1.f;
            float heightScale = 1.f;
            float3 absorptionColor = float3(0.35f, 0.25f, 0.15f);

            float baseWeightScale = 1.f;
            float midWeightScale = 1.f;
            float coatWeightScale = 1.f;

            float baseNormalFlatten = 0.f;
            float midNormalFlatten = 0.35f;
            float coatNormalFlatten = 0.7f;

            uint32_t enableBaseLayer = 1;
            uint32_t enableMidLayer = 1;
            uint32_t enableCoatLayer = 1;
            uint32_t flipNormalY = 1;
            uint32_t showThickness = 0;
        };
        static_assert(sizeof(Data) <= sizeof(MaterialPayload), "LayeredTextureMaterial payload must fit in MaterialPayload");

        void setupTextureSlots();
        ref<Texture> loadExrTexture(const std::filesystem::path& path, bool singleChannel) const;
        static void renderTextureInfo(Gui::Widgets& widget, const char* label, const ref<Texture>& pTexture);

        Data mData = {};
        ref<Sampler> mpSampler;
    };
}
