#include "MaterialXGraphMaterial.h"

#include "Utils/Logger.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <utility>

namespace Falcor
{
namespace
{
using json = nlohmann::json;

std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

TextureFilteringMode parseFilterMode(const std::string& v)
{
    const std::string value = toLower(v);
    return value == "point" ? TextureFilteringMode::Point : TextureFilteringMode::Linear;
}

TextureAddressingMode parseAddressMode(const std::string& v)
{
    const std::string value = toLower(v);
    if (value == "mirror") return TextureAddressingMode::Mirror;
    if (value == "clamp") return TextureAddressingMode::Clamp;
    if (value == "border") return TextureAddressingMode::Border;
    if (value == "mirroronce") return TextureAddressingMode::MirrorOnce;
    return TextureAddressingMode::Wrap;
}

bool parseBool(const json& j, bool defaultValue = false)
{
    return j.is_boolean() ? j.get<bool>() : defaultValue;
}

uint32_t parseUInt(const json& j, uint32_t defaultValue = 0)
{
    return j.is_number_unsigned() ? j.get<uint32_t>() : j.is_number_integer() ? (uint32_t)j.get<int32_t>() : defaultValue;
}

float parseFloat(const json& j, float defaultValue = 0.f)
{
    return j.is_number() ? j.get<float>() : defaultValue;
}

std::string parseString(const json& j, const char* key, std::string defaultValue = {})
{
    auto it = j.find(key);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : std::move(defaultValue);
}

std::optional<float4> parseFloatArray(const json& j)
{
    if (!j.is_array()) return std::nullopt;
    const size_t n = j.size();
    if (n < 1 || n > 4) return std::nullopt;

    float4 v(0.f);
    for (size_t i = 0; i < n; ++i)
    {
        if (!j[i].is_number()) return std::nullopt;
        v[i] = j[i].get<float>();
    }
    return v;
}
} // namespace

MaterialType MaterialXGraphMaterial::registerGeneratedMaterialType(const std::string& typeName)
{
    FALCOR_CHECK(!typeName.empty(), "MaterialXGraphMaterial requires a non-empty generated type name.");
    return registerMaterialType(typeName);
}

MaterialXGraphMaterial::MaterialXGraphMaterial(
    ref<Device> pDevice,
    const std::string& name,
    const std::filesystem::path& modulePath,
    const std::string& typeName,
    const std::filesystem::path& manifestPath
)
    : Material(pDevice, name, registerGeneratedMaterialType(typeName))
    , mModulePath(modulePath)
    , mTypeName(typeName)
    , mManifestPath(manifestPath)
{
    loadManifest();
    loadTextureBindings();
    applyKnownMaterialMetadata();
    markUpdates(UpdateFlags::DataChanged | UpdateFlags::ResourcesChanged | UpdateFlags::CodeChanged);
}

bool MaterialXGraphMaterial::renderUI(Gui::Widgets& widget)
{
    widget.text("Generated type: " + mTypeName);
    widget.text("Module: " + mModulePath.string());
    widget.text("Manifest: " + mManifestPath.string());
    widget.text(fmt::format("Generated textures: {}", mTextureBindings.size()));

    for (size_t i = 0; i < mTextureBindings.size(); ++i)
    {
        const auto& b = mTextureBindings[i];
        widget.text(fmt::format("[{}] {} -> {}", i, b.semantic.empty() ? "<untyped>" : b.semantic, b.sourcePath.string()));
    }

    return false;
}

Material::UpdateFlags MaterialXGraphMaterial::update(MaterialSystem* pOwner)
{
    (void)pOwner;
    return std::exchange(mUpdates, UpdateFlags::None);
}

bool MaterialXGraphMaterial::isEqual(const ref<Material>& pOther) const
{
    auto p = dynamic_ref_cast<MaterialXGraphMaterial>(pOther);
    if (!p) return false;
    if (!isBaseEqual(*p)) return false;
    if (mModulePath != p->mModulePath || mTypeName != p->mTypeName || mManifestPath != p->mManifestPath) return false;
    if (mTextureBindings.size() != p->mTextureBindings.size()) return false;
    if (mConstantBindings.size() != p->mConstantBindings.size()) return false;

    for (size_t i = 0; i < mTextureBindings.size(); ++i)
    {
        const auto& a = mTextureBindings[i];
        const auto& b = p->mTextureBindings[i];
        if (a.semantic != b.semantic || a.sourcePath != b.sourcePath || a.shaderTextureName != b.shaderTextureName ||
            a.shaderSamplerName != b.shaderSamplerName || a.srgb != b.srgb || a.generateMips != b.generateMips || a.pTexture != b.pTexture ||
            a.pSampler != b.pSampler)
        {
            return false;
        }
    }

    for (size_t i = 0; i < mConstantBindings.size(); ++i)
    {
        const auto& a = mConstantBindings[i];
        const auto& b = p->mConstantBindings[i];
        if (a.shaderName != b.shaderName || a.kind != b.kind || a.boolValue != b.boolValue || a.intValue != b.intValue ||
            a.uintValue != b.uintValue || a.floatValue.x != b.floatValue.x || a.floatValue.y != b.floatValue.y ||
                a.floatValue.z != b.floatValue.z || a.floatValue.w != b.floatValue.w)
        {
            return false;
        }
    }

    return true;
}

ProgramDesc::ShaderModuleList MaterialXGraphMaterial::getShaderModules() const
{
    return {ProgramDesc::ShaderModule::fromFile(mModulePath.string())};
}

TypeConformanceList MaterialXGraphMaterial::getTypeConformances() const
{
    return {{{mTypeName, "IMaterial"}, (uint32_t)getType()}};
}


void MaterialXGraphMaterial::loadManifest()
{
    FALCOR_CHECK(std::filesystem::exists(mManifestPath), "MaterialX manifest '{}' does not exist.", mManifestPath.string());

    std::ifstream f(mManifestPath);
    FALCOR_CHECK(f.good(), "Failed to open MaterialX manifest '{}'.", mManifestPath.string());

    json manifest;
    f >> manifest;

    if (mTypeName.empty())
    {
        if (auto it = manifest.find("typeName"); it != manifest.end() && it->is_string()) mTypeName = it->get<std::string>();
    }

    if (auto it = manifest.find("modulePath"); it != manifest.end() && it->is_string() && mModulePath.empty())
    {
        mModulePath = resolveManifestRelativePath(mManifestPath, it->get<std::string>());
    }

    if (auto it = manifest.find("alphaMode"); it != manifest.end() && it->is_string())
    {
        const std::string mode = toLower(it->get<std::string>());

        // Falcor only supports Opaque and Mask here.
        if (mode == "mask")
        {
            setAlphaMode(AlphaMode::Mask);
        }
        else
        {
            // "blend" is not a valid Falcor AlphaMode in this API.
            // Treat it as opaque for now.
            setAlphaMode(AlphaMode::Opaque);
        }
    }

    if (auto it = manifest.find("alphaThreshold"); it != manifest.end() && it->is_number()) setAlphaThreshold(it->get<float>());
    if (auto it = manifest.find("doubleSided"); it != manifest.end()) setDoubleSided(parseBool(*it));
    if (auto it = manifest.find("thinSurface"); it != manifest.end()) setThinSurface(parseBool(*it));
    if (auto it = manifest.find("ior"); it != manifest.end() && it->is_number()) setIndexOfRefraction(it->get<float>());

    if (auto it = manifest.find("textures"); it != manifest.end() && it->is_array())
    {
        for (const auto& jtex : *it)
        {
            TextureBinding b;
            b.semantic = parseString(jtex, "semantic");
            b.sourcePath = resolveManifestRelativePath(mManifestPath, parseString(jtex, "file"));
            b.shaderTextureName = parseString(jtex, "shaderTextureName");
            b.shaderSamplerName = parseString(jtex, "shaderSamplerName");
            const std::string cs = toLower(parseString(jtex, "colorSpace"));
                b.srgb = (cs == "srgb" || cs == "srgb_texture" || cs == "srgbtexture" || jtex.value("srgb", false));
            b.generateMips = jtex.contains("generateMipLevels") ? parseBool(jtex["generateMipLevels"], true) : true;

            if (jtex.contains("sampler") && jtex["sampler"].is_object())
            {
                const json& js = jtex["sampler"];
                const std::string filter = parseString(js, "filter", "linear");
                const std::string addressMode = parseString(js, "addressMode", "wrap");
                const uint32_t maxAniso = js.value("maxAnisotropy", 1u);
                b.pSampler = createSampler(mpDevice, filter, addressMode, maxAniso);
            }
            else
            {
                b.pSampler = createSampler(mpDevice);
            }

            mTextureBindings.push_back(std::move(b));
        }
    }

    auto parseConstantValue = [](const json& value, ConstantBinding& binding) -> bool
    {
        if (value.is_boolean())
        {
            binding.kind = ConstantBinding::Kind::Bool;
            binding.boolValue = value.get<bool>();
            return true;
        }
        if (value.is_number_unsigned())
        {
            binding.kind = ConstantBinding::Kind::UInt;
            binding.uintValue = value.get<uint32_t>();
            return true;
        }
        if (value.is_number_integer())
        {
            binding.kind = ConstantBinding::Kind::Int;
            binding.intValue = value.get<int32_t>();
            return true;
        }
        if (value.is_number_float())
        {
            binding.kind = ConstantBinding::Kind::Float;
            binding.floatValue.x = value.get<float>();
            return true;
        }
        if (auto v = parseFloatArray(value); v.has_value())
        {
            binding.floatValue = *v;
            switch (value.size())
            {
            case 2: binding.kind = ConstantBinding::Kind::Float2; break;
            case 3: binding.kind = ConstantBinding::Kind::Float3; break;
            case 4: binding.kind = ConstantBinding::Kind::Float4; break;
            default: binding.kind = ConstantBinding::Kind::Float; break;
            }
            return true;
        }
        return false;
    };

    if (auto it = manifest.find("constants"); it != manifest.end())
    {
        if (it->is_object())
        {
            for (auto kv = it->begin(); kv != it->end(); ++kv)
            {
                ConstantBinding binding;
                binding.shaderName = kv.key();
                if (!parseConstantValue(kv.value(), binding)) continue;
                mConstantBindings.push_back(std::move(binding));
            }
        }
        else if (it->is_array())
        {
            for (const auto& jc : *it)
            {
                if (!jc.is_object()) continue;

                ConstantBinding binding;
                binding.shaderName = parseString(jc, "shaderVariableName");
                if (binding.shaderName.empty()) binding.shaderName = parseString(jc, "shaderName");
                if (binding.shaderName.empty()) continue;

                auto valIt = jc.find("value");
                if (valIt == jc.end()) continue;
                if (!parseConstantValue(*valIt, binding)) continue;
                mConstantBindings.push_back(std::move(binding));
            }
        }
    }
}

void MaterialXGraphMaterial::loadTextureBindings()
{
    for (auto& binding : mTextureBindings)
    {
        if (binding.sourcePath.empty()) continue;

        binding.pTexture = Texture::createFromFile(mpDevice, binding.sourcePath, binding.generateMips, binding.srgb);
        if (!binding.pTexture) continue;

        bool slotValid = false;
        const TextureSlot slot = tryGetTextureSlotFromSemantic(binding.semantic, slotValid);
        if (slotValid)
        {
            mTextureSlotData[(size_t)slot].pTexture = binding.pTexture;
        }
    }
}

void MaterialXGraphMaterial::applyKnownMaterialMetadata()
{
    for (const auto& binding : mTextureBindings)
    {
        bool valid = false;
        const TextureSlot slot = tryGetTextureSlotFromSemantic(binding.semantic, valid);
        if (!valid) continue;

        TextureSlotInfo info = {};
        info.name = binding.semantic.empty() ? to_string(slot) : binding.semantic;
        info.srgb = binding.srgb;

        switch (slot)
        {
        case TextureSlot::BaseColor:
            info.mask = TextureChannelFlags::RGBA;
            break;
        case TextureSlot::Specular:
            info.mask = TextureChannelFlags::RGBA;
            break;
        case TextureSlot::Emissive:
            info.mask = TextureChannelFlags::RGB;
            break;
        case TextureSlot::Normal:
            info.mask = TextureChannelFlags::RGB;
            info.srgb = false;
            break;
        case TextureSlot::Transmission:
            info.mask = TextureChannelFlags::RGB;
            break;
        case TextureSlot::Displacement:
            info.mask = TextureChannelFlags::RGB;
            info.srgb = false;
            break;
        default:
            info.mask = TextureChannelFlags::RGBA;
            break;
        }

        mTextureSlotInfo[(size_t)slot] = info;
    }
}

ref<Sampler> MaterialXGraphMaterial::createSampler(ref<Device> pDevice)
{
    Sampler::Desc desc;
    desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    desc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    return pDevice->createSampler(desc);
}

ref<Sampler> MaterialXGraphMaterial::createSampler(ref<Device> pDevice, const std::string& filter, const std::string& addressMode, uint32_t maxAnisotropy)
{
    Sampler::Desc desc;
    const auto filterMode = parseFilterMode(filter);
    const auto addrMode = parseAddressMode(addressMode);
    desc.setFilterMode(filterMode, filterMode, filterMode);
    desc.setAddressingMode(addrMode, addrMode, addrMode);
    desc.setMaxAnisotropy(std::max(1u, maxAnisotropy));
    return pDevice->createSampler(desc);
}

std::filesystem::path MaterialXGraphMaterial::resolveManifestRelativePath(
    const std::filesystem::path& manifestPath,
    const std::filesystem::path& relativePath
)
{
    if (relativePath.empty()) return {};
    if (relativePath.is_absolute()) return relativePath;
    return manifestPath.parent_path() / relativePath;
}

Material::TextureSlot MaterialXGraphMaterial::tryGetTextureSlotFromSemantic(const std::string& semantic, bool& valid)
{
    const std::string s = toLower(semantic);
    valid = true;

    if (s == "basecolor" || s == "base_color" || s == "albedo" || s == "diffuse") return TextureSlot::BaseColor;
    if (s == "specular" || s == "metallicroughness" || s == "metalrough" || s == "metallic" || s == "roughness") return TextureSlot::Specular;
    if (s == "normal") return TextureSlot::Normal;
    if (s == "emissive" || s == "emission") return TextureSlot::Emissive;
    if (s == "transmission") return TextureSlot::Transmission;
    if (s == "displacement" || s == "height") return TextureSlot::Displacement;

    valid = false;
    return TextureSlot::Count;
}
} // namespace Falcor
