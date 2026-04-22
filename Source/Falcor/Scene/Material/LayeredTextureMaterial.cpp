#include "LayeredTextureMaterial.h"

#include "MaterialSystem.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
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
        const std::string kShaderFile = "Scene/Material/LayeredTextureMaterial.slang";

        MaterialType getLayeredTextureMaterialType()
        {
            static MaterialType sType = registerMaterialType("LayeredTextureMaterial");
            return sType;
        }
    }

    LayeredTextureMaterial::LayeredTextureMaterial(ref<Device> pDevice, const std::string& name)
        : Material(pDevice, name, getLayeredTextureMaterialType())
    {
        setupTextureSlots();
        markUpdates(UpdateFlags::DataChanged | UpdateFlags::ResourcesChanged | UpdateFlags::CodeChanged);
    }

    LayeredTextureMaterial::LayeredTextureMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        : LayeredTextureMaterial(std::move(pDevice), name)
    {
        loadTextureSet(textureDirectory);
    }

    void LayeredTextureMaterial::setupTextureSlots()
    {
        mTextureSlotInfo[(uint32_t)TextureSlot::BaseColor] = {"baseColor", TextureChannelFlags::RGB, true};
        mTextureSlotInfo[(uint32_t)TextureSlot::Specular] = {"roughness", TextureChannelFlags::Red, false};
        mTextureSlotInfo[(uint32_t)TextureSlot::Normal] = {"normal", TextureChannelFlags::RGB, false};
        mTextureSlotInfo[(uint32_t)TextureSlot::Displacement] = {"displacement", TextureChannelFlags::Red, false};
        mTextureSlotInfo[(uint32_t)TextureSlot::Emissive] = {"layerRoughness", TextureChannelFlags::RGB, false};
        mTextureSlotInfo[(uint32_t)TextureSlot::Transmission] = {"layerWeights", TextureChannelFlags::RGB, false};
    }

    void LayeredTextureMaterial::renderTextureInfo(Gui::Widgets& widget, const char* label, const ref<Texture>& pTexture)
    {
        if (!pTexture) return;

        widget.text(std::string(label) + ": " + pTexture->getSourcePath().string());
        widget.text(
            "Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" +
            to_string(pTexture->getFormat()) + ")"
        );
        widget.image(label, pTexture.get(), float2(100.f));
    }

    ref<Texture> LayeredTextureMaterial::loadExrTexture(const std::filesystem::path& path, bool singleChannel) const
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
            logWarning("LayeredTextureMaterial: OpenEXR fallback failed for '{}': {}", path.string(), e.what());
            return nullptr;
        }
    }

    bool LayeredTextureMaterial::renderUI(Gui::Widgets& widget)
    {
        bool dirty = Material::renderUI(widget);

        renderTextureInfo(widget, "Base color", getBaseColorTexture());
        renderTextureInfo(widget, "Roughness", getRoughnessTexture());
        renderTextureInfo(widget, "Normal", getNormalTexture());
        renderTextureInfo(widget, "Displacement", getDisplacementTexture());
        renderTextureInfo(widget, "Packed layer roughness", getLayerRoughnessTexture());
        renderTextureInfo(widget, "Packed layer weights", getLayerWeightTexture());

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

        bool showThickness = mData.showThickness != 0;
        if (widget.checkbox("Show thickness debug", showThickness))
        {
            mData.showThickness = showThickness ? 1u : 0u;
            markUpdates(UpdateFlags::DataChanged);
        }

        if (widget.var("Base F0", mData.baseF0, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Mid F0", mData.midF0, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Coat F0", mData.coatF0, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Roughness scale", mData.roughnessScale, 0.f, 4.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Roughness bias", mData.roughnessBias, -1.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Base weight scale", mData.baseWeightScale, 0.f, 4.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Mid weight scale", mData.midWeightScale, 0.f, 4.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Coat weight scale", mData.coatWeightScale, 0.f, 4.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Base normal flatten", mData.baseNormalFlatten, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Mid normal flatten", mData.midNormalFlatten, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Coat normal flatten", mData.coatNormalFlatten, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Height scale", mData.heightScale, 0.f, 100.f, 0.1f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Thickness scale", mData.thicknessScale, 0.f, 100.f, 0.1f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Absorption color", mData.absorptionColor, 0.f, 10.f, 0.05f)) markUpdates(UpdateFlags::DataChanged);

        dirty |= mUpdates != UpdateFlags::None;
        return dirty;
    }

    Material::UpdateFlags LayeredTextureMaterial::update(MaterialSystem* pOwner)
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
        updateTextureHandle(pOwner, TextureSlot::Specular, mData.texRoughness);
        updateTextureHandle(pOwner, TextureSlot::Normal, mData.texNormal);
        updateTextureHandle(pOwner, TextureSlot::Displacement, mData.texDisplacement);
        updateTextureHandle(pOwner, TextureSlot::Emissive, mData.texLayerRoughness);
        updateTextureHandle(pOwner, TextureSlot::Transmission, mData.texLayerWeights);
        updateDefaultTextureSamplerID(pOwner, mpSampler);

        UpdateFlags updates = mUpdates;
        mUpdates = UpdateFlags::None;
        return updates;
    }

    bool LayeredTextureMaterial::isEqual(const ref<Material>& pOther) const
    {
        auto p = dynamic_ref_cast<LayeredTextureMaterial>(pOther);
        if (!p) return false;
        return isBaseEqual(*p) &&
               mData.baseF0 == p->mData.baseF0 &&
               mData.midF0 == p->mData.midF0 &&
               mData.coatF0 == p->mData.coatF0 &&
               mData.roughnessScale == p->mData.roughnessScale &&
               mData.roughnessBias == p->mData.roughnessBias &&
               mData.thicknessScale == p->mData.thicknessScale &&
               mData.heightScale == p->mData.heightScale &&
               all(mData.absorptionColor == p->mData.absorptionColor) &&
               mData.baseWeightScale == p->mData.baseWeightScale &&
               mData.midWeightScale == p->mData.midWeightScale &&
               mData.coatWeightScale == p->mData.coatWeightScale &&
               mData.baseNormalFlatten == p->mData.baseNormalFlatten &&
               mData.midNormalFlatten == p->mData.midNormalFlatten &&
               mData.coatNormalFlatten == p->mData.coatNormalFlatten &&
               mData.enableBaseLayer == p->mData.enableBaseLayer &&
               mData.enableMidLayer == p->mData.enableMidLayer &&
               mData.enableCoatLayer == p->mData.enableCoatLayer &&
               mData.flipNormalY == p->mData.flipNormalY &&
               mData.showThickness == p->mData.showThickness &&
               getBaseColorTexture() == p->getBaseColorTexture() &&
               getRoughnessTexture() == p->getRoughnessTexture() &&
               getNormalTexture() == p->getNormalTexture() &&
               getDisplacementTexture() == p->getDisplacementTexture() &&
               getLayerRoughnessTexture() == p->getLayerRoughnessTexture() &&
               getLayerWeightTexture() == p->getLayerWeightTexture() &&
               mpSampler == p->mpSampler;
    }

    bool LayeredTextureMaterial::loadTextureSet(const std::filesystem::path& textureDirectory)
    {
        if (textureDirectory.empty()) return false;

        auto load = [&](TextureSlot slot, const char* label, const std::filesystem::path& path, bool srgb)
        {
            logInfo("LayeredTextureMaterial: loading {} texture '{}'.", label, path.string());
            if (!std::filesystem::exists(path))
            {
                logWarning("LayeredTextureMaterial: missing {} texture '{}'.", label, path.string());
                return false;
            }

            ref<Texture> pTexture;
            if (hasExtension(path, "exr"))
            {
                pTexture = loadExrTexture(path, slot == TextureSlot::Specular || slot == TextureSlot::Displacement);
            }
            else
            {
                pTexture = Texture::createFromFile(mpDevice, path, true, srgb);
            }

            if (!pTexture)
            {
                logWarning("LayeredTextureMaterial: failed to load {} texture '{}'.", label, path.string());
                return false;
            }

            setTexture(slot, pTexture);
            logInfo(
                "LayeredTextureMaterial: loaded {} texture '{}' as {}x{} {}.",
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
            logWarning("LayeredTextureMaterial: no candidate file found for {}.", label);
            return false;
        };

        bool loaded = false;
        loaded |= load(TextureSlot::BaseColor, "baseColor", textureDirectory / "rough_concrete_diff_8k.jpg", true);
        loaded |= load(TextureSlot::Specular, "roughness", textureDirectory / "rough_concrete_rough_8k.exr", false);
        loaded |= load(TextureSlot::Normal, "normal", textureDirectory / "rough_concrete_nor_gl_8k.exr", false);
        loaded |= load(TextureSlot::Displacement, "displacement", textureDirectory / "rough_concrete_disp_8k.png", false);

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
            TextureSlot::Transmission,
            "packed layer weights",
            {
                textureDirectory / "layer_weights_packed_8k.exr",
                textureDirectory / "layer_weights_packed_8k.png",
                textureDirectory / "layered_weights_packed_8k.exr",
                textureDirectory / "layered_weights_packed_8k.png"
            },
            false
        );

        logInfo("LayeredTextureMaterial: texture set load {}.", loaded ? "completed" : "did not load any textures");
        return loaded;
    }

    MaterialDataBlob LayeredTextureMaterial::getDataBlob() const
    {
        return prepareDataBlob(mData);
    }

    ProgramDesc::ShaderModuleList LayeredTextureMaterial::getShaderModules() const
    {
        return {ProgramDesc::ShaderModule::fromFile(kShaderFile)};
    }

    TypeConformanceList LayeredTextureMaterial::getTypeConformances() const
    {
        return {{{"LayeredTextureMaterial", "IMaterial"}, (uint32_t)getType()}};
    }
}
