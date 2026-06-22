#include "ThreeLayeredGGXNoHeightMaterial.h"

#include "MaterialSystem.h"
#include "Core/API/Device.h"
#include "GlobalState.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/StringUtils.h"

#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>

#include <limits>
#include <utility>

namespace Falcor
{
    namespace
    {
        const std::string kShaderFile = "Scene/Material/ThreeLayeredGGXNoHeightMaterial.slang";
        const std::vector<std::string> kBootstrapFeatureNames = {
            "base_color.r",
            "base_color.g",
            "base_color.b",
            "base_roughness",
            "mid_roughness",
            "coat_roughness",
            "base_normal.x",
            "base_normal.y",
            "base_normal.z",
            "base_tangent.x",
            "base_tangent.y",
            "base_tangent.z",
            "mid_normal.x",
            "mid_normal.y",
            "mid_normal.z",
            "mid_tangent.x",
            "mid_tangent.y",
            "mid_tangent.z",
            "coat_normal.x",
            "coat_normal.y",
            "coat_normal.z",
            "coat_tangent.x",
            "coat_tangent.y",
            "coat_tangent.z",
            "dust_coverage",
        };

        MaterialType getThreeLayeredGGXNoHeightMaterialType()
        {
            static MaterialType sType = registerMaterialType("ThreeLayeredGGXNoHeightMaterial");
            return sType;
        }
    }

    ThreeLayeredGGXNoHeightMaterial::ThreeLayeredGGXNoHeightMaterial(ref<Device> pDevice, const std::string& name)
        : Material(pDevice, name, getThreeLayeredGGXNoHeightMaterialType())
    {
        setupTextureSlots();
        markUpdates(UpdateFlags::DataChanged | UpdateFlags::ResourcesChanged | UpdateFlags::CodeChanged);
    }

    ThreeLayeredGGXNoHeightMaterial::ThreeLayeredGGXNoHeightMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        : ThreeLayeredGGXNoHeightMaterial(std::move(pDevice), name)
    {
        loadTextureSet(textureDirectory);
    }

    void ThreeLayeredGGXNoHeightMaterial::setupTextureSlots()
    {
        mTextureSlotInfo[(uint32_t)TextureSlot::BaseColor] = {"baseColor", TextureChannelFlags::RGB, true};
        mTextureSlotInfo[(uint32_t)TextureSlot::Normal] = {"normal", TextureChannelFlags::RGB, false};
        mTextureSlotInfo[(uint32_t)TextureSlot::Emissive] = {"layerRoughness", TextureChannelFlags::RGB, false};
        mTextureSlotInfo[(uint32_t)TextureSlot::Index] = {"dustCoverage", TextureChannelFlags::Red, false};
    }

    void ThreeLayeredGGXNoHeightMaterial::renderTextureInfo(Gui::Widgets& widget, const char* label, const ref<Texture>& pTexture)
    {
        if (!pTexture) return;

        widget.text(std::string(label) + ": " + pTexture->getSourcePath().string());
        widget.text(
            "Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" +
            to_string(pTexture->getFormat()) + ")"
        );
        widget.image(label, pTexture.get(), float2(100.f));
    }

    ref<Texture> ThreeLayeredGGXNoHeightMaterial::loadExrTexture(const std::filesystem::path& path, bool singleChannel) const
    {
        try
        {
            Imf::InputFile file(path.string().c_str());
            const Imf::Header& header = file.header();
            const Imath::Box2i& dataWindow = header.dataWindow();
            const uint32_t width = uint32_t(dataWindow.max.x - dataWindow.min.x + 1);
            const uint32_t height = uint32_t(dataWindow.max.y - dataWindow.min.y + 1);
            const size_t pixelCount = size_t(width) * size_t(height);

            FALCOR_CHECK(width > 0 && height > 0, "EXR '{}' has invalid dimensions.", path.string());

            const Imf::ChannelList& channels = header.channels();
            auto findChannelName = [&](std::initializer_list<const char*> names)
            {
                for (const char* name : names)
                {
                    if (channels.findChannel(name) != nullptr) return std::string(name);
                }
                return std::string();
            };

            auto findFirstChannelName = [&]()
            {
                auto it = channels.begin();
                return it == channels.end() ? std::string() : std::string(it.name());
            };

            std::string rName = findChannelName({"R", "r", "Y", "y"});
            if (rName.empty()) rName = findFirstChannelName();
            std::string gName = findChannelName({"G", "g"});
            std::string bName = findChannelName({"B", "b"});
            FALCOR_CHECK(!rName.empty(), "EXR '{}' does not contain any image channels.", path.string());
            if (gName.empty()) gName = rName;
            if (bName.empty()) bName = rName;

            Imf::FrameBuffer frameBuffer;
            if (singleChannel)
            {
                std::vector<uint16_t> pixels(pixelCount, 0);
                char* base = reinterpret_cast<char*>(pixels.data()) -
                             (dataWindow.min.x + dataWindow.min.y * int(width)) * ptrdiff_t(sizeof(uint16_t));
                frameBuffer.insert(rName.c_str(), Imf::Slice(Imf::HALF, base, sizeof(uint16_t), sizeof(uint16_t) * width));

                file.setFrameBuffer(frameBuffer);
                file.readPixels(dataWindow.min.y, dataWindow.max.y);

                auto pTexture = mpDevice->createTexture2D(
                    width,
                    height,
                    ResourceFormat::R16Float,
                    1,
                    1,
                    pixels.data(),
                    ResourceBindFlags::ShaderResource
                );
                if (pTexture) pTexture->setSourcePath(path);
                return pTexture;
            }

            std::vector<uint16_t> pixels(pixelCount * 4, math::float16_t(0.f).toBits());
            const uint16_t one = math::float16_t(1.f).toBits();
            for (size_t i = 0; i < pixelCount; ++i) pixels[i * 4 + 3] = one;

            const size_t xStride = sizeof(uint16_t) * 4;
            const size_t yStride = xStride * width;
            char* base = reinterpret_cast<char*>(pixels.data()) -
                         (dataWindow.min.x + dataWindow.min.y * int(width)) * ptrdiff_t(xStride);

            frameBuffer.insert(rName.c_str(), Imf::Slice(Imf::HALF, base + sizeof(uint16_t) * 0, xStride, yStride));
            frameBuffer.insert(gName.c_str(), Imf::Slice(Imf::HALF, base + sizeof(uint16_t) * 1, xStride, yStride));
            frameBuffer.insert(bName.c_str(), Imf::Slice(Imf::HALF, base + sizeof(uint16_t) * 2, xStride, yStride));

            file.setFrameBuffer(frameBuffer);
            file.readPixels(dataWindow.min.y, dataWindow.max.y);

            auto pTexture = mpDevice->createTexture2D(
                width,
                height,
                ResourceFormat::RGBA16Float,
                1,
                1,
                pixels.data(),
                ResourceBindFlags::ShaderResource
            );
            if (pTexture) pTexture->setSourcePath(path);
            return pTexture;
        }
        catch (const std::exception& e)
        {
            logWarning("ThreeLayeredGGXNoHeightMaterial: OpenEXR fallback failed for '{}': {}", path.string(), e.what());
            return nullptr;
        }
    }

    bool ThreeLayeredGGXNoHeightMaterial::renderUI(Gui::Widgets& widget)
    {
        bool dirty = Material::renderUI(widget);

        renderTextureInfo(widget, "Base color", getBaseColorTexture());
        renderTextureInfo(widget, "Normal", getNormalTexture());
        renderTextureInfo(widget, "Packed layer roughness", getLayerRoughnessTexture());
        renderTextureInfo(widget, "Dust coverage", getDustCoverageTexture());

        bool enableBaseLayer = mData.enableBaseLayer != 0;
        if (widget.checkbox("Enable base layer", enableBaseLayer))
        {
            mData.enableBaseLayer = enableBaseLayer ? 1u : 0u;
            markUpdates(UpdateFlags::DataChanged);
        }

        bool enableMidLayer = mData.enableMidLayer != 0;
        if (widget.checkbox("Enable mid layer", enableMidLayer))
        {
            mData.enableMidLayer = enableMidLayer ? 1u : 0u;
            markUpdates(UpdateFlags::DataChanged);
        }

        bool enableCoatLayer = mData.enableCoatLayer != 0;
        if (widget.checkbox("Enable coat layer", enableCoatLayer))
        {
            mData.enableCoatLayer = enableCoatLayer ? 1u : 0u;
            markUpdates(UpdateFlags::DataChanged);
        }

        bool flipNormalY = mData.flipNormalY != 0;
        if (widget.checkbox("Flip normal Y (OpenGL normal map)", flipNormalY))
        {
            mData.flipNormalY = flipNormalY ? 1u : 0u;
            markUpdates(UpdateFlags::DataChanged);
        }

        if (widget.var("Base F0", mData.baseF0, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Mid F0", mData.midF0, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Coat F0", mData.coatF0, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Roughness scale", mData.roughnessScale, 0.f, 4.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Roughness bias", mData.roughnessBias, -1.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Dust coverage scale", mData.dustCoverageScale, 0.f, 4.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Base normal flatten", mData.baseNormalFlatten, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Mid normal flatten", mData.midNormalFlatten, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Coat normal flatten", mData.coatNormalFlatten, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);

        dirty |= mUpdates != UpdateFlags::None;
        return dirty;
    }

    Material::UpdateFlags ThreeLayeredGGXNoHeightMaterial::update(MaterialSystem* pOwner)
    {
        FALCOR_ASSERT(pOwner);

        if (!mpSampler)
        {
            Sampler::Desc desc;
            desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
            desc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
            mpSampler = mpDevice->createSampler(desc);
            mUpdates |= UpdateFlags::ResourcesChanged;
        }

        updateTextureHandle(pOwner, TextureSlot::BaseColor, mData.texBaseColor);
        updateTextureHandle(pOwner, TextureSlot::Normal, mData.texNormal);
        updateTextureHandle(pOwner, TextureSlot::Emissive, mData.texLayerRoughness);
        updateTextureHandle(pOwner, TextureSlot::Index, mData.texDustCoverage);
        updateDefaultTextureSamplerID(pOwner, mpSampler);

        UpdateFlags updates = mUpdates;
        mUpdates = UpdateFlags::None;
        return updates;
    }

    bool ThreeLayeredGGXNoHeightMaterial::isEqual(const ref<Material>& pOther) const
    {
        auto p = dynamic_ref_cast<ThreeLayeredGGXNoHeightMaterial>(pOther);
        if (!p) return false;
        return isBaseEqual(*p) &&
               mData.baseF0 == p->mData.baseF0 &&
               mData.midF0 == p->mData.midF0 &&
               mData.coatF0 == p->mData.coatF0 &&
               mData.roughnessScale == p->mData.roughnessScale &&
               mData.roughnessBias == p->mData.roughnessBias &&
               mData.dustCoverageScale == p->mData.dustCoverageScale &&
               mData.baseNormalFlatten == p->mData.baseNormalFlatten &&
               mData.midNormalFlatten == p->mData.midNormalFlatten &&
               mData.coatNormalFlatten == p->mData.coatNormalFlatten &&
               mData.enableBaseLayer == p->mData.enableBaseLayer &&
               mData.enableMidLayer == p->mData.enableMidLayer &&
               mData.enableCoatLayer == p->mData.enableCoatLayer &&
               mData.flipNormalY == p->mData.flipNormalY &&
               getBaseColorTexture() == p->getBaseColorTexture() &&
               getNormalTexture() == p->getNormalTexture() &&
               getLayerRoughnessTexture() == p->getLayerRoughnessTexture() &&
               getDustCoverageTexture() == p->getDustCoverageTexture() &&
               mpSampler == p->mpSampler;
    }

    bool ThreeLayeredGGXNoHeightMaterial::loadTextureSet(const std::filesystem::path& textureDirectory)
    {
        if (textureDirectory.empty()) return false;

        auto load = [&](TextureSlot slot, const char* label, const std::filesystem::path& path, bool srgb)
        {
            logInfo("ThreeLayeredGGXNoHeightMaterial: loading {} texture '{}'.", label, path.string());
            if (!std::filesystem::exists(path))
            {
                logWarning("ThreeLayeredGGXNoHeightMaterial: missing {} texture '{}'.", label, path.string());
                return false;
            }

            ref<Texture> pTexture;
            if (hasExtension(path, "exr"))
            {
                pTexture = loadExrTexture(path, slot == TextureSlot::Specular);
            }
            else
            {
                pTexture = Texture::createFromFile(mpDevice, path, true, srgb);
            }

            if (!pTexture)
            {
                logWarning("ThreeLayeredGGXNoHeightMaterial: failed to load {} texture '{}'.", label, path.string());
                return false;
            }

            setTexture(slot, pTexture);
            logInfo(
                "ThreeLayeredGGXNoHeightMaterial: loaded {} texture '{}' as {}x{} {}.",
                label,
                path.filename().string(),
                pTexture->getWidth(),
                pTexture->getHeight(),
                to_string(pTexture->getFormat())
            );
            return true;
        };

        auto loadFirstExisting = [&](TextureSlot slot, const char* label, std::initializer_list<std::filesystem::path> candidates, bool srgb)
        {
            for (const auto& candidate : candidates)
            {
                if (std::filesystem::exists(candidate)) return load(slot, label, candidate, srgb);
            }
            logWarning("ThreeLayeredGGXNoHeightMaterial: no candidate file found for {}.", label);
            return false;
        };

        bool loaded = false;
        loaded |= load(TextureSlot::BaseColor, "baseColor", textureDirectory / "rough_concrete_diff_8k.jpg", true);
        loaded |= load(TextureSlot::Normal, "normal", textureDirectory / "rough_concrete_nor_gl_8k.exr", false);

        loadFirstExisting(
            TextureSlot::Emissive,
            "packed layer roughness",
            {
                textureDirectory / "layer_roughness_packed_8k.exr",
                textureDirectory / "layer_roughness_packed_8k.png",
                textureDirectory / "layered_roughness_packed_8k.exr",
                textureDirectory / "layered_roughness_packed_8k.png"
            },
            false
        );

        loadFirstExisting(
            TextureSlot::Index,
            "dust coverage",
            {
                textureDirectory / "dust_coverage_8k.png",
                textureDirectory / "dust_coverage_8k.exr"
            },
            false
        );

        logInfo("ThreeLayeredGGXNoHeightMaterial: texture set load {}.", loaded ? "completed" : "did not load any textures");
        return loaded;
    }

    MaterialDataBlob ThreeLayeredGGXNoHeightMaterial::getDataBlob() const
    {
        return prepareDataBlob(mData);
    }

    ProgramDesc::ShaderModuleList ThreeLayeredGGXNoHeightMaterial::getShaderModules() const
    {
        return {ProgramDesc::ShaderModule::fromFile(kShaderFile)};
    }

    TypeConformanceList ThreeLayeredGGXNoHeightMaterial::getTypeConformances() const
    {
        TypeConformanceList conformances;
        conformances.add("ThreeLayeredGGXNoHeightMaterial", "IMaterial", (uint32_t)getType());
        conformances.add("ThreeLayeredGGXNoHeightMaterial", "IBootstrapFeatureMaterial", (uint32_t)getType());
        conformances.add("ThreeLayeredGGXNoHeightMaterial", "ITrainingDirectionGuardMaterial", (uint32_t)getType());
        return conformances;
    }

    std::vector<std::string> ThreeLayeredGGXNoHeightMaterial::getBootstrapFeatureNames() const
    {
        return kBootstrapFeatureNames;
    }

    FALCOR_SCRIPT_BINDING(ThreeLayeredGGXNoHeightMaterial)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)

        pybind11::class_<ThreeLayeredGGXNoHeightMaterial, Material, ref<ThreeLayeredGGXNoHeightMaterial>> material(m, "ThreeLayeredGGXNoHeightMaterial");
        auto create = [](const std::string& name, const std::filesystem::path& textureDirectory)
        {
            auto resolvedPath = getActiveAssetResolver().resolvePath(textureDirectory);
            FALCOR_CHECK(!resolvedPath.empty(), "Layered texture directory '{}' could not be resolved.", textureDirectory.string());
            FALCOR_CHECK(std::filesystem::is_directory(resolvedPath), "Layered texture path '{}' is not a directory.", resolvedPath.string());

            auto pMaterial = ThreeLayeredGGXNoHeightMaterial::create(accessActivePythonSceneBuilder().getDevice(), name);
            pMaterial->loadTextureSet(resolvedPath);
            return pMaterial;
        };
        material.def(
            pybind11::init(create),
            "name"_a,
            "textureDirectory"_a
        );
    }
}
