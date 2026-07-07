#pragma once

#include "Material.h"
#include "Core/API/Sampler.h"
#include "Scene/Material/MaterialTypeRegistry.h"

#include <filesystem>
#include <utility>

namespace Falcor
{
    class ThreeLayeredGGXBaseOnlyMaterial : public Material
    {
        FALCOR_OBJECT(ThreeLayeredGGXBaseOnlyMaterial)

    public:
        static ref<ThreeLayeredGGXBaseOnlyMaterial> create(ref<Device> pDevice, const std::string& name = "")
        {
            return make_ref<ThreeLayeredGGXBaseOnlyMaterial>(std::move(pDevice), name);
        }

        static ref<ThreeLayeredGGXBaseOnlyMaterial> create(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        {
            return make_ref<ThreeLayeredGGXBaseOnlyMaterial>(std::move(pDevice), name, textureDirectory);
        }

        ThreeLayeredGGXBaseOnlyMaterial(ref<Device> pDevice, const std::string& name);
        ThreeLayeredGGXBaseOnlyMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory);

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

        void setLayerRoughnessTexture(const ref<Texture>& pTexture) { setTexture(TextureSlot::Emissive, pTexture); }
        ref<Texture> getLayerRoughnessTexture() const { return getTexture(TextureSlot::Emissive); }

        bool loadTextureSet(const std::filesystem::path& textureDirectory);
        void setBaseF0(float f0);
        float getBaseF0() const { return mData.baseF0; }

    private:
        struct Data
        {
            TextureHandle texBaseColor;
            TextureHandle texNormal;
            TextureHandle texLayerRoughness;

            float baseF0 = 0.02f;
            float roughnessScale = 1.f;
            float roughnessBias = 0.02f;
            float normalFlatten = 0.f;

            uint32_t flipNormalY = 1;
        };
        static_assert(sizeof(Data) <= sizeof(MaterialPayload), "ThreeLayeredGGXBaseOnlyMaterial payload must fit in MaterialPayload");

        void setupTextureSlots();
        ref<Texture> loadExrTexture(const std::filesystem::path& path, bool singleChannel) const;
        static void renderTextureInfo(Gui::Widgets& widget, const char* label, const ref<Texture>& pTexture);

        Data mData = {};
        ref<Sampler> mpSampler;
    };
}
