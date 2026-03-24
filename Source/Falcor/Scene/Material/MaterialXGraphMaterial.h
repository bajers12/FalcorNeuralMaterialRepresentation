#pragma once

#include "Scene/Material/Material.h"
#include "Core/Program/ShaderVar.h"

#include <filesystem>
#include <string>
#include <vector>

namespace Falcor
{
class MaterialXGraphMaterial : public Material
{
public:
    FALCOR_OBJECT(MaterialXGraphMaterial)

    static ref<MaterialXGraphMaterial> create(
        ref<Device> pDevice,
        const std::string& name,
        const std::filesystem::path& modulePath,
        const std::string& typeName,
        const std::filesystem::path& manifestPath
    )
    {
        return make_ref<MaterialXGraphMaterial>(pDevice, name, modulePath, typeName, manifestPath);
    }

    MaterialXGraphMaterial(
        ref<Device> pDevice,
        const std::string& name,
        const std::filesystem::path& modulePath,
        const std::string& typeName,
        const std::filesystem::path& manifestPath
    );

    bool renderUI(Gui::Widgets& widget) override;
    UpdateFlags update(MaterialSystem* pOwner) override;
    bool isEqual(const ref<Material>& pOther) const override;

    MaterialDataBlob getDataBlob() const override { return prepareDataBlob(mData); }
    ProgramDesc::ShaderModuleList getShaderModules() const override;
    TypeConformanceList getTypeConformances() const override;

    const std::filesystem::path& getModulePath() const { return mModulePath; }
    const std::string& getGeneratedTypeName() const { return mTypeName; }
    const std::filesystem::path& getManifestPath() const { return mManifestPath; }

    /// Binds textures/samplers/constants described in the generated MaterialX manifest.
    /// Call this after creating ProgramVars/ParameterBlocks for a program that includes this material.
    void bindGeneratedResources(const ShaderVar& var) const
    {
        for (const auto& binding : mTextureBindings)
        {
            if (!binding.shaderTextureName.empty())
            {
                const ShaderVar texVar = var.findMember(binding.shaderTextureName);
                if (texVar.isValid() && binding.pTexture) texVar.setTexture(binding.pTexture);
            }

            if (!binding.shaderSamplerName.empty())
            {
                const ShaderVar samplerVar = var.findMember(binding.shaderSamplerName);
                if (samplerVar.isValid() && binding.pSampler) samplerVar.setSampler(binding.pSampler);
            }
        }

        for (const auto& binding : mConstantBindings)
        {
            const ShaderVar c = var.findMember(binding.shaderName);
            if (!c.isValid()) continue;

            switch (binding.kind)
            {
            case ConstantBinding::Kind::Bool:
                c = binding.boolValue;
                break;
            case ConstantBinding::Kind::Int:
                c = binding.intValue;
                break;
            case ConstantBinding::Kind::UInt:
                c = binding.uintValue;
                break;
            case ConstantBinding::Kind::Float:
                c = binding.floatValue.x;
                break;
            case ConstantBinding::Kind::Float2:
                c = binding.floatValue.xy();
                break;
            case ConstantBinding::Kind::Float3:
                c = binding.floatValue.xyz();
                break;
            case ConstantBinding::Kind::Float4:
                c = binding.floatValue;
                break;
            }
        }
    }   

private:
    struct Data
    {
        uint32_t flags = 0;
        uint32_t reserved0 = 0;
        uint32_t reserved1 = 0;
        uint32_t reserved2 = 0;
    };

    struct TextureBinding
    {
        std::string semantic;
        std::filesystem::path sourcePath;
        std::string shaderTextureName;
        std::string shaderSamplerName;
        bool srgb = false;
        bool generateMips = true;
        ref<Texture> pTexture;
        ref<Sampler> pSampler;
    };

    struct ConstantBinding
    {
        std::string shaderName;
        enum class Kind
        {
            Bool,
            Int,
            UInt,
            Float,
            Float2,
            Float3,
            Float4,
        } kind = Kind::Float;

        bool boolValue = false;
        int32_t intValue = 0;
        uint32_t uintValue = 0;
        float4 floatValue = float4(0.f);
    };

    static MaterialType registerGeneratedMaterialType(const std::string& typeName);

    void loadManifest();
    void loadTextureBindings();
    void applyKnownMaterialMetadata();

    static ref<Sampler> createSampler(ref<Device> pDevice);
    static ref<Sampler> createSampler(ref<Device> pDevice, const std::string& filter, const std::string& addressMode, uint32_t maxAnisotropy);
    static std::filesystem::path resolveManifestRelativePath(const std::filesystem::path& manifestPath, const std::filesystem::path& relativePath);
    static Material::TextureSlot tryGetTextureSlotFromSemantic(const std::string& semantic, bool& valid);

    Data mData = {};
    std::filesystem::path mModulePath;
    std::string mTypeName;
    std::filesystem::path mManifestPath;

    std::vector<TextureBinding> mTextureBindings;
    std::vector<ConstantBinding> mConstantBindings;
};
} // namespace Falcor
