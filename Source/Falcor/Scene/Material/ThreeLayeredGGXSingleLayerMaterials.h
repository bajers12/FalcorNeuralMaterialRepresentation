#pragma once

#include "Material.h"
#include "Core/API/Sampler.h"
#include "Scene/Material/MaterialTypeRegistry.h"

#include <filesystem>
#include <string>
#include <utility>

namespace Falcor
{
    class ThreeLayeredGGXSingleLayerMaterialBase : public Material
    {
    public:
        bool renderUI(Gui::Widgets& widget) override;
        UpdateFlags update(MaterialSystem* pOwner) override;
        bool isEqual(const ref<Material>& pOther) const override;
        MaterialDataBlob getDataBlob() const override;
        ProgramDesc::ShaderModuleList getShaderModules() const override;
        TypeConformanceList getTypeConformances() const override;
        std::vector<std::string> getBootstrapFeatureNames() const override;

        void setDefaultTextureSampler(const ref<Sampler>& pSampler) override { mpSampler = pSampler; }
        ref<Sampler> getDefaultTextureSampler() const override { return mpSampler; }

        bool loadTextureSet(const std::filesystem::path& textureDirectory);
        void setF0(float f0);
        float getF0() const { return mData.f0; }

        void setNormalFlatten(float flatten);
        float getNormalFlatten() const { return mData.normalFlatten; }

    protected:
        struct Data
        {
            TextureHandle texBaseColor;
            TextureHandle texNormal;
            TextureHandle texLayerRoughness;
            TextureHandle texDustCoverage;

            float f0 = 0.2f;
            float roughnessScale = 1.f;
            float roughnessBias = 0.02f;
            float normalFlatten = 0.f;

            uint32_t flipNormalY = 1;
        };
        static_assert(sizeof(Data) <= sizeof(MaterialPayload), "Single layer material payload must fit in MaterialPayload");

        ThreeLayeredGGXSingleLayerMaterialBase(
            ref<Device> pDevice,
            const std::string& name,
            MaterialType materialType,
            std::string materialLabel,
            std::string shaderFile,
            std::string slangTypeName,
            std::vector<std::string> bootstrapFeatureNames
        );

        ThreeLayeredGGXSingleLayerMaterialBase(
            ref<Device> pDevice,
            const std::string& name,
            MaterialType materialType,
            std::string materialLabel,
            std::string shaderFile,
            std::string slangTypeName,
            std::vector<std::string> bootstrapFeatureNames,
            const std::filesystem::path& textureDirectory
        );

        void setupTextureSlots();
        ref<Texture> loadExrTexture(const std::filesystem::path& path, bool singleChannel) const;
        static void renderTextureInfo(Gui::Widgets& widget, const char* label, const ref<Texture>& pTexture);

        Data mData = {};
        ref<Sampler> mpSampler;
        std::string mMaterialLabel;
        std::string mShaderFile;
        std::string mSlangTypeName;
        std::vector<std::string> mBootstrapFeatureNames;
    };

    class ThreeLayeredGGXMidOnlyMaterial : public ThreeLayeredGGXSingleLayerMaterialBase
    {
        FALCOR_OBJECT(ThreeLayeredGGXMidOnlyMaterial)
    public:
        static ref<ThreeLayeredGGXMidOnlyMaterial> create(ref<Device> pDevice, const std::string& name = "")
        {
            return make_ref<ThreeLayeredGGXMidOnlyMaterial>(std::move(pDevice), name);
        }
        static ref<ThreeLayeredGGXMidOnlyMaterial> create(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        {
            return make_ref<ThreeLayeredGGXMidOnlyMaterial>(std::move(pDevice), name, textureDirectory);
        }
        ThreeLayeredGGXMidOnlyMaterial(ref<Device> pDevice, const std::string& name);
        ThreeLayeredGGXMidOnlyMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory);
    };

    class ThreeLayeredGGXCoatOnlyMaterial : public ThreeLayeredGGXSingleLayerMaterialBase
    {
        FALCOR_OBJECT(ThreeLayeredGGXCoatOnlyMaterial)
    public:
        static ref<ThreeLayeredGGXCoatOnlyMaterial> create(ref<Device> pDevice, const std::string& name = "")
        {
            return make_ref<ThreeLayeredGGXCoatOnlyMaterial>(std::move(pDevice), name);
        }
        static ref<ThreeLayeredGGXCoatOnlyMaterial> create(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        {
            return make_ref<ThreeLayeredGGXCoatOnlyMaterial>(std::move(pDevice), name, textureDirectory);
        }
        ThreeLayeredGGXCoatOnlyMaterial(ref<Device> pDevice, const std::string& name);
        ThreeLayeredGGXCoatOnlyMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory);
    };

    class ThreeLayeredGGXDustOnlyMaterial : public ThreeLayeredGGXSingleLayerMaterialBase
    {
        FALCOR_OBJECT(ThreeLayeredGGXDustOnlyMaterial)
    public:
        static ref<ThreeLayeredGGXDustOnlyMaterial> create(ref<Device> pDevice, const std::string& name = "")
        {
            return make_ref<ThreeLayeredGGXDustOnlyMaterial>(std::move(pDevice), name);
        }
        static ref<ThreeLayeredGGXDustOnlyMaterial> create(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        {
            return make_ref<ThreeLayeredGGXDustOnlyMaterial>(std::move(pDevice), name, textureDirectory);
        }
        ThreeLayeredGGXDustOnlyMaterial(ref<Device> pDevice, const std::string& name);
        ThreeLayeredGGXDustOnlyMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory);
    };
}
