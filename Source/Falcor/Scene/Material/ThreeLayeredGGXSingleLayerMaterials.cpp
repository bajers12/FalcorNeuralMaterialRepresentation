#include "ThreeLayeredGGXSingleLayerMaterials.h"

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

#include <algorithm>
#include <utility>

namespace Falcor
{
    namespace
    {
        MaterialType getMidOnlyMaterialType()
        {
            static MaterialType sType = registerMaterialType("ThreeLayeredGGXMidOnlyMaterial");
            return sType;
        }

        MaterialType getCoatOnlyMaterialType()
        {
            static MaterialType sType = registerMaterialType("ThreeLayeredGGXCoatOnlyMaterial");
            return sType;
        }

        MaterialType getDustOnlyMaterialType()
        {
            static MaterialType sType = registerMaterialType("ThreeLayeredGGXDustOnlyMaterial");
            return sType;
        }

        const std::vector<std::string>& specularFeatureNames(const char* prefix)
        {
            static const std::vector<std::string> midNames = {
                "mid_roughness", "mid_normal.x", "mid_normal.y", "mid_normal.z", "mid_tangent.x", "mid_tangent.y", "mid_tangent.z"
            };
            static const std::vector<std::string> coatNames = {
                "coat_roughness", "coat_normal.x", "coat_normal.y", "coat_normal.z", "coat_tangent.x", "coat_tangent.y", "coat_tangent.z"
            };
            return std::string(prefix) == "coat" ? coatNames : midNames;
        }

        const std::vector<std::string>& dustFeatureNames()
        {
            static const std::vector<std::string> names = {
                "dust_coverage", "dust_normal.x", "dust_normal.y", "dust_normal.z", "dust_tangent.x", "dust_tangent.y", "dust_tangent.z"
            };
            return names;
        }
    }

    ThreeLayeredGGXSingleLayerMaterialBase::ThreeLayeredGGXSingleLayerMaterialBase(
        ref<Device> pDevice,
        const std::string& name,
        MaterialType materialType,
        std::string materialLabel,
        std::string shaderFile,
        std::string slangTypeName,
        std::vector<std::string> bootstrapFeatureNames
    )
        : Material(pDevice, name, materialType)
        , mMaterialLabel(std::move(materialLabel))
        , mShaderFile(std::move(shaderFile))
        , mSlangTypeName(std::move(slangTypeName))
        , mBootstrapFeatureNames(std::move(bootstrapFeatureNames))
    {
        setupTextureSlots();
        markUpdates(UpdateFlags::DataChanged | UpdateFlags::ResourcesChanged | UpdateFlags::CodeChanged);
    }

    ThreeLayeredGGXSingleLayerMaterialBase::ThreeLayeredGGXSingleLayerMaterialBase(
        ref<Device> pDevice,
        const std::string& name,
        MaterialType materialType,
        std::string materialLabel,
        std::string shaderFile,
        std::string slangTypeName,
        std::vector<std::string> bootstrapFeatureNames,
        const std::filesystem::path& textureDirectory
    )
        : ThreeLayeredGGXSingleLayerMaterialBase(
              std::move(pDevice),
              name,
              materialType,
              std::move(materialLabel),
              std::move(shaderFile),
              std::move(slangTypeName),
              std::move(bootstrapFeatureNames)
          )
    {
        loadTextureSet(textureDirectory);
    }

    void ThreeLayeredGGXSingleLayerMaterialBase::setupTextureSlots()
    {
        mTextureSlotInfo[(uint32_t)TextureSlot::BaseColor] = {"baseColor", TextureChannelFlags::RGB, true};
        mTextureSlotInfo[(uint32_t)TextureSlot::Normal] = {"normal", TextureChannelFlags::RGB, false};
        mTextureSlotInfo[(uint32_t)TextureSlot::Emissive] = {"layerRoughness", TextureChannelFlags::RGB, false};
        mTextureSlotInfo[(uint32_t)TextureSlot::Index] = {"dustCoverage", TextureChannelFlags::Red, false};
    }

    void ThreeLayeredGGXSingleLayerMaterialBase::renderTextureInfo(Gui::Widgets& widget, const char* label, const ref<Texture>& pTexture)
    {
        if (!pTexture) return;
        widget.text(std::string(label) + ": " + pTexture->getSourcePath().string());
        widget.text(
            "Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" +
            to_string(pTexture->getFormat()) + ")"
        );
        widget.image(label, pTexture.get(), float2(100.f));
    }

    ref<Texture> ThreeLayeredGGXSingleLayerMaterialBase::loadExrTexture(const std::filesystem::path& path, bool singleChannel) const
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
                    if (channels.findChannel(name) != nullptr) return std::string(name);
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

                auto pTexture = mpDevice->createTexture2D(width, height, ResourceFormat::R16Float, 1, 1, pixels.data(), ResourceBindFlags::ShaderResource);
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

            auto pTexture = mpDevice->createTexture2D(width, height, ResourceFormat::RGBA16Float, 1, 1, pixels.data(), ResourceBindFlags::ShaderResource);
            if (pTexture) pTexture->setSourcePath(path);
            return pTexture;
        }
        catch (const std::exception& e)
        {
            logWarning("{}: OpenEXR fallback failed for '{}': {}", mMaterialLabel, path.string(), e.what());
            return nullptr;
        }
    }

    bool ThreeLayeredGGXSingleLayerMaterialBase::loadTextureSet(const std::filesystem::path& textureDirectory)
    {
        if (textureDirectory.empty()) return false;

        auto load = [&](TextureSlot slot, const char* label, const std::filesystem::path& path, bool srgb, bool singleChannelExr = false)
        {
            if (!std::filesystem::exists(path)) return false;
            ref<Texture> pTexture = hasExtension(path, "exr") ? loadExrTexture(path, singleChannelExr) : Texture::createFromFile(mpDevice, path, true, srgb);
            if (!pTexture)
            {
                logWarning("{}: failed to load {} texture '{}'.", mMaterialLabel, label, path.string());
                return false;
            }
            setTexture(slot, pTexture);
            logInfo("{}: loaded {} texture '{}' as {}x{} {}.", mMaterialLabel, label, path.filename().string(), pTexture->getWidth(), pTexture->getHeight(), to_string(pTexture->getFormat()));
            return true;
        };

        auto loadFirstExisting = [&](TextureSlot slot, const char* label, std::initializer_list<std::filesystem::path> candidates, bool srgb, bool singleChannelExr = false)
        {
            for (const auto& candidate : candidates)
                if (load(slot, label, candidate, srgb, singleChannelExr)) return true;
            logWarning("{}: no candidate file found for {}.", mMaterialLabel, label);
            return false;
        };

        bool loaded = false;
        loaded |= load(TextureSlot::BaseColor, "baseColor", textureDirectory / "rough_concrete_diff_8k.jpg", true);
        loaded |= load(TextureSlot::Normal, "normal", textureDirectory / "rough_concrete_nor_gl_8k.exr", false);
        loaded |= loadFirstExisting(
            TextureSlot::Emissive,
            "packed layer roughness",
            {textureDirectory / "layer_roughness_packed_8k.exr", textureDirectory / "layer_roughness_packed_8k.png"},
            false
        );
        loaded |= loadFirstExisting(
            TextureSlot::Index,
            "dust coverage",
            {textureDirectory / "dust_coverage_8k.exr", textureDirectory / "dust_coverage_8k.png"},
            false,
            true
        );

        return loaded;
    }

    bool ThreeLayeredGGXSingleLayerMaterialBase::renderUI(Gui::Widgets& widget)
    {
        bool dirty = Material::renderUI(widget);
        renderTextureInfo(widget, "Base color", getTexture(TextureSlot::BaseColor));
        renderTextureInfo(widget, "Normal", getTexture(TextureSlot::Normal));
        renderTextureInfo(widget, "Packed layer roughness", getTexture(TextureSlot::Emissive));
        renderTextureInfo(widget, "Dust coverage", getTexture(TextureSlot::Index));

        bool flipNormalY = mData.flipNormalY != 0;
        if (widget.checkbox("Flip normal Y (OpenGL normal map)", flipNormalY))
        {
            mData.flipNormalY = flipNormalY ? 1u : 0u;
            markUpdates(UpdateFlags::DataChanged);
        }
        if (widget.var("F0", mData.f0, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Roughness scale", mData.roughnessScale, 0.f, 4.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Roughness bias", mData.roughnessBias, -1.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);
        if (widget.var("Normal flatten", mData.normalFlatten, 0.f, 1.f, 0.01f)) markUpdates(UpdateFlags::DataChanged);

        dirty |= mUpdates != UpdateFlags::None;
        return dirty;
    }

    Material::UpdateFlags ThreeLayeredGGXSingleLayerMaterialBase::update(MaterialSystem* pOwner)
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

    bool ThreeLayeredGGXSingleLayerMaterialBase::isEqual(const ref<Material>& pOther) const
    {
        auto p = dynamic_ref_cast<ThreeLayeredGGXSingleLayerMaterialBase>(pOther);
        if (!p) return false;
        return isBaseEqual(*p) &&
               getType() == p->getType() &&
               mData.f0 == p->mData.f0 &&
               mData.roughnessScale == p->mData.roughnessScale &&
               mData.roughnessBias == p->mData.roughnessBias &&
               mData.normalFlatten == p->mData.normalFlatten &&
               mData.flipNormalY == p->mData.flipNormalY;
    }

    MaterialDataBlob ThreeLayeredGGXSingleLayerMaterialBase::getDataBlob() const
    {
        return prepareDataBlob(mData);
    }

    ProgramDesc::ShaderModuleList ThreeLayeredGGXSingleLayerMaterialBase::getShaderModules() const
    {
        return {ProgramDesc::ShaderModule::fromFile(mShaderFile)};
    }

    TypeConformanceList ThreeLayeredGGXSingleLayerMaterialBase::getTypeConformances() const
    {
        TypeConformanceList conformances;
        conformances.add(mSlangTypeName, "IMaterial", (uint32_t)getType());
        conformances.add(mSlangTypeName, "IBootstrapFeatureMaterial", (uint32_t)getType());
        conformances.add(mSlangTypeName, "ITrainingDirectionGuardMaterial", (uint32_t)getType());
        return conformances;
    }

    std::vector<std::string> ThreeLayeredGGXSingleLayerMaterialBase::getBootstrapFeatureNames() const
    {
        return mBootstrapFeatureNames;
    }

    void ThreeLayeredGGXSingleLayerMaterialBase::setF0(float f0)
    {
        float clamped = std::clamp(f0, 0.f, 1.f);
        if (mData.f0 == clamped) return;
        mData.f0 = clamped;
        markUpdates(UpdateFlags::DataChanged);
    }

    void ThreeLayeredGGXSingleLayerMaterialBase::setNormalFlatten(float flatten)
    {
        float clamped = std::clamp(flatten, 0.f, 1.f);
        if (mData.normalFlatten == clamped) return;
        mData.normalFlatten = clamped;
        markUpdates(UpdateFlags::DataChanged);
    }

    ThreeLayeredGGXMidOnlyMaterial::ThreeLayeredGGXMidOnlyMaterial(ref<Device> pDevice, const std::string& name)
        : ThreeLayeredGGXSingleLayerMaterialBase(
              std::move(pDevice),
              name,
              getMidOnlyMaterialType(),
              "ThreeLayeredGGXMidOnlyMaterial",
              "Scene/Material/ThreeLayeredGGXMidOnlyMaterial.slang",
              "ThreeLayeredGGXMidOnlyMaterial",
              specularFeatureNames("mid")
          )
    {
        mData.f0 = 0.2f;
        mData.normalFlatten = 0.2f;
    }

    ThreeLayeredGGXMidOnlyMaterial::ThreeLayeredGGXMidOnlyMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        : ThreeLayeredGGXMidOnlyMaterial(std::move(pDevice), name)
    {
        loadTextureSet(textureDirectory);
    }

    ThreeLayeredGGXCoatOnlyMaterial::ThreeLayeredGGXCoatOnlyMaterial(ref<Device> pDevice, const std::string& name)
        : ThreeLayeredGGXSingleLayerMaterialBase(
              std::move(pDevice),
              name,
              getCoatOnlyMaterialType(),
              "ThreeLayeredGGXCoatOnlyMaterial",
              "Scene/Material/ThreeLayeredGGXCoatOnlyMaterial.slang",
              "ThreeLayeredGGXCoatOnlyMaterial",
              specularFeatureNames("coat")
          )
    {
        mData.f0 = 0.3f;
        mData.normalFlatten = 0.8f;
    }

    ThreeLayeredGGXCoatOnlyMaterial::ThreeLayeredGGXCoatOnlyMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        : ThreeLayeredGGXCoatOnlyMaterial(std::move(pDevice), name)
    {
        loadTextureSet(textureDirectory);
    }

    ThreeLayeredGGXDustOnlyMaterial::ThreeLayeredGGXDustOnlyMaterial(ref<Device> pDevice, const std::string& name)
        : ThreeLayeredGGXSingleLayerMaterialBase(
              std::move(pDevice),
              name,
              getDustOnlyMaterialType(),
              "ThreeLayeredGGXDustOnlyMaterial",
              "Scene/Material/ThreeLayeredGGXDustOnlyMaterial.slang",
              "ThreeLayeredGGXDustOnlyMaterial",
              dustFeatureNames()
          )
    {
        mData.f0 = 0.f;
        mData.normalFlatten = 0.8f;
    }

    ThreeLayeredGGXDustOnlyMaterial::ThreeLayeredGGXDustOnlyMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& textureDirectory)
        : ThreeLayeredGGXDustOnlyMaterial(std::move(pDevice), name)
    {
        loadTextureSet(textureDirectory);
    }

    template<typename T>
    void bindSingleLayerMaterial(pybind11::module& m, const char* name)
    {
        using namespace pybind11::literals;
        pybind11::class_<T, ThreeLayeredGGXSingleLayerMaterialBase, ref<T>> material(m, name);
        auto create = [](const std::string& materialName, const std::filesystem::path& textureDirectory)
        {
            auto resolvedPath = getActiveAssetResolver().resolvePath(textureDirectory);
            FALCOR_CHECK(!resolvedPath.empty(), "Layered texture directory '{}' could not be resolved.", textureDirectory.string());
            FALCOR_CHECK(std::filesystem::is_directory(resolvedPath), "Layered texture path '{}' is not a directory.", resolvedPath.string());
            auto pMaterial = T::create(accessActivePythonSceneBuilder().getDevice(), materialName);
            pMaterial->loadTextureSet(resolvedPath);
            return pMaterial;
        };
        material.def(pybind11::init(create), "name"_a, "textureDirectory"_a);
        material.def_property("f0", &T::getF0, &T::setF0);
        material.def_property("normalFlatten", &T::getNormalFlatten, &T::setNormalFlatten);
    }

    FALCOR_SCRIPT_BINDING(ThreeLayeredGGXSingleLayerMaterials)
    {
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)

        pybind11::class_<ThreeLayeredGGXSingleLayerMaterialBase, Material, ref<ThreeLayeredGGXSingleLayerMaterialBase>>(
            m, "ThreeLayeredGGXSingleLayerMaterialBase"
        );
        bindSingleLayerMaterial<ThreeLayeredGGXMidOnlyMaterial>(m, "ThreeLayeredGGXMidOnlyMaterial");
        bindSingleLayerMaterial<ThreeLayeredGGXCoatOnlyMaterial>(m, "ThreeLayeredGGXCoatOnlyMaterial");
        bindSingleLayerMaterial<ThreeLayeredGGXDustOnlyMaterial>(m, "ThreeLayeredGGXDustOnlyMaterial");
    }
}
