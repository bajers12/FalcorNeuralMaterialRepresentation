#include "NeuralMaterial.h"
#include "MaterialSystem.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
#include <algorithm>
#include <fstream>

namespace Falcor
{
    namespace
    {
        constexpr const char kWeightMagic02[8] = { 'N','M','D','L','W','T','0','2' };
        const std::string kShaderFile = "Scene/Material/NeuralMaterial.slang";
    }
    namespace
    {
        MaterialType getNeuralMaterialType()
        {
            static MaterialType sType = registerMaterialType("NeuralMaterial");
            return sType;
        }
    }

    NeuralMaterial::NeuralMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& basePath)
        : Material(pDevice, name, getNeuralMaterialType())
        , mBasePath(basePath)
    {
        loadAssets();
        markUpdates(UpdateFlags::DataChanged | UpdateFlags::ResourcesChanged | UpdateFlags::CodeChanged);
    }

    void NeuralMaterial::setBasePath(const std::filesystem::path& path)
    {
        if (mBasePath != path)
        {
            mBasePath = path;
            loadAssets();
            markUpdates(UpdateFlags::DataChanged | UpdateFlags::ResourcesChanged | UpdateFlags::CodeChanged);
        }
    }

    bool NeuralMaterial::renderUI(Gui::Widgets& widget)
    {
        bool dirty = Material::renderUI(widget);
        widget.text("Neural asset path: " + mBasePath.string());
        widget.text("Latent mip levels: " + std::to_string(mLatentMipCount));
        dirty |= widget.checkbox("Force latent mip", mForceLatentMip);
        uint32_t maxMip = mLatentMipCount > 0 ? mLatentMipCount - 1 : 0;
        dirty |= widget.var("Forced latent mip", mForcedLatentMip, 0u, maxMip);
        dirty |= widget.var("Latent mip debug mode", mLatentMipDebugMode, 0u, 3u);
        bool samplerDirty = widget.var("Latent LOD bias", mLatentLodBias, -16.f, 16.f, 0.01f);
        dirty |= samplerDirty;
        if (dirty)
        {
            mForcedLatentMip = std::min(mForcedLatentMip, maxMip);
            mLatentMipDebugMode = std::min(mLatentMipDebugMode, 3u);
            mData.latentMipControl = packLatentMipControl();
            markUpdates(UpdateFlags::DataChanged);
            if (samplerDirty)
            {
                mpSampler = createLatentSampler();
                markUpdates(UpdateFlags::ResourcesChanged);
            }
        }
        return dirty;
    }

    ref<Sampler> NeuralMaterial::createLatentSampler() const
    {
        Sampler::Desc desc;
        desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        desc.setMaxAnisotropy(8);
        desc.setLodParams(-1000.f, 1000.f, mLatentLodBias);
        return mpDevice->createSampler(desc);
    }

    uint32_t NeuralMaterial::packLatentMipControl() const
    {
        uint32_t mipCount = std::clamp(mLatentMipCount, 1u, 255u);
        uint32_t forcedMip = std::clamp(mForcedLatentMip, 0u, mipCount - 1u);
        uint32_t debugMode = std::clamp(mLatentMipDebugMode, 0u, 3u);

        return (mForceLatentMip ? (1u << 31) : 0u) |
               (debugMode << 16) |
               (forcedMip << 8) |
               mipCount;
    }

    std::vector<float> NeuralMaterial::readFloatArray(std::ifstream& f, size_t count)
    {
        std::vector<float> v(count);
        f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
        if (!f) FALCOR_THROW("Failed reading float array from decoder_weights.bin");
        return v;
    }

    void NeuralMaterial::loadAssets()
    {
        const auto latent0Path = mBasePath / "latent0.exr";
        const auto latent1Path = mBasePath / "latent1.exr";
        const auto weightsPath = mBasePath / "decoder_weights.bin";
        const auto samplerWeightsPath = mBasePath / "sampler_weights.bin";

        std::vector<std::filesystem::path> latent0MipPaths;
        std::vector<std::filesystem::path> latent1MipPaths;
        for (uint32_t mip = 0; mip < 255; ++mip)
        {
            auto p0 = mBasePath / ("latent0_mip" + std::to_string(mip) + ".exr");
            auto p1 = mBasePath / ("latent1_mip" + std::to_string(mip) + ".exr");
            if (!std::filesystem::exists(p0) || !std::filesystem::exists(p1))
                break;
            latent0MipPaths.push_back(p0);
            latent1MipPaths.push_back(p1);
        }

        if (!latent0MipPaths.empty())
        {
            mpLatent0 = Texture::createMippedFromFiles(mpDevice, latent0MipPaths, false);
            mpLatent1 = Texture::createMippedFromFiles(mpDevice, latent1MipPaths, false);
            mLatentMipCount = static_cast<uint32_t>(latent0MipPaths.size());
        }
        else
        {
            mpLatent0 = Texture::createFromFile(mpDevice, latent0Path.string(), false, false);
            mpLatent1 = Texture::createFromFile(mpDevice, latent1Path.string(), false, false);
            mLatentMipCount = 1;
        }

        if (!mpLatent0) FALCOR_THROW("Failed to load latent texture: {}", latent0Path.string());
        if (!mpLatent1) FALCOR_THROW("Failed to load latent texture: {}", latent1Path.string());
        mForcedLatentMip = std::min(mForcedLatentMip, mLatentMipCount - 1);
        mData.latentMipControl = packLatentMipControl();

        auto makeStructured = [&](const std::vector<float>& data) -> ref<Buffer>
        {
            return make_ref<Buffer>(
                mpDevice,
                sizeof(float),
                static_cast<uint32_t>(data.size()),
                ResourceBindFlags::ShaderResource,
                MemoryType::DeviceLocal,
                data.data(),
                false
            );
        };

        auto loadDecoderWeights = [&](const std::filesystem::path& path, uint32_t expectedInputDim, uint32_t expectedOutputDim, bool hasFrameLinear)
        {
            std::ifstream f(path, std::ios::binary);
            if (!f) FALCOR_THROW("Failed to open weight file: {}", path.string());

            char magic[8];
            f.read(magic, 8);
            if (!f)
                FALCOR_THROW("Invalid weight file magic in: {}", path.string());

            if (std::memcmp(magic, kWeightMagic02, 8) != 0)
                FALCOR_THROW("Invalid weight file magic in: {}", path.string());

            int32_t latentCh = 8;
            int32_t numFrames = 2;

            int32_t mlpWidth = 32;
            int32_t mlpDepth = 2;

            f.read(reinterpret_cast<char*>(&latentCh), sizeof(int32_t));
            f.read(reinterpret_cast<char*>(&numFrames), sizeof(int32_t));
            f.read(reinterpret_cast<char*>(&mlpWidth), sizeof(int32_t));
            f.read(reinterpret_cast<char*>(&mlpDepth), sizeof(int32_t));
            if (!f) FALCOR_THROW("Failed reading weight file header: {}", path.string());

            if (latentCh != 8) FALCOR_THROW("Expected latentCh == 8, got {} in {}", latentCh, path.string());
            if (hasFrameLinear && numFrames != 2)
                FALCOR_THROW("Expected numFrames == 2 for frame-based decoder, got {} in {}", numFrames, path.string());
            if (mlpWidth != 16 && mlpWidth != 32 && mlpWidth != 64)
                FALCOR_THROW("Expected mlpWidth in {16, 32, 64}, got {} in {}", mlpWidth, path.string());
            if (mlpDepth != 2 && mlpDepth != 3)
                FALCOR_THROW("Expected mlpDepth in {2, 3}, got {} in {}", mlpDepth, path.string());

            std::vector<float> frameLinear;
            if (hasFrameLinear)
                frameLinear = readFloatArray(f, 12 * 8);
            auto w0 = readFloatArray(f, static_cast<size_t>(mlpWidth) * expectedInputDim);
            auto b0 = readFloatArray(f, static_cast<size_t>(mlpWidth));
            auto w1 = readFloatArray(f, static_cast<size_t>(mlpWidth) * static_cast<size_t>(mlpWidth));
            auto b1 = readFloatArray(f, static_cast<size_t>(mlpWidth));
            std::vector<float> w2;
            std::vector<float> b2;
            std::vector<float> w3;
            std::vector<float> b3;

            if (mlpDepth == 2)
            {
                w2 = readFloatArray(f, static_cast<size_t>(expectedOutputDim) * static_cast<size_t>(mlpWidth));
                b2 = readFloatArray(f, expectedOutputDim);
            }
            else
            {
                w2 = readFloatArray(f, static_cast<size_t>(mlpWidth) * static_cast<size_t>(mlpWidth));
                b2 = readFloatArray(f, static_cast<size_t>(mlpWidth));
                w3 = readFloatArray(f, static_cast<size_t>(expectedOutputDim) * static_cast<size_t>(mlpWidth));
                b3 = readFloatArray(f, expectedOutputDim);
            }

            // Pack all weights into a single buffer
            std::vector<float> packedData;
            Data::DecoderWeightOffsets offsets{};

            // Frame linear weights (BRDF only).
            offsets.frameLinearOffset = static_cast<uint32_t>(packedData.size());
            if (hasFrameLinear)
                packedData.insert(packedData.end(), frameLinear.begin(), frameLinear.end());

            // Layer 0 weights and bias
            offsets.w0Offset = static_cast<uint32_t>(packedData.size());
            packedData.insert(packedData.end(), w0.begin(), w0.end());

            offsets.b0Offset = static_cast<uint32_t>(packedData.size());
            packedData.insert(packedData.end(), b0.begin(), b0.end());

            // Layer 1 weights and bias
            offsets.w1Offset = static_cast<uint32_t>(packedData.size());
            packedData.insert(packedData.end(), w1.begin(), w1.end());

            offsets.b1Offset = static_cast<uint32_t>(packedData.size());
            packedData.insert(packedData.end(), b1.begin(), b1.end());

            // Layer 2 weights and bias
            offsets.w2Offset = static_cast<uint32_t>(packedData.size());
            packedData.insert(packedData.end(), w2.begin(), w2.end());

            offsets.b2Offset = static_cast<uint32_t>(packedData.size());
            packedData.insert(packedData.end(), b2.begin(), b2.end());

            // Layer 3 weights and bias (if depth == 3)
            offsets.w3Offset = static_cast<uint32_t>(packedData.size());
            if (!w3.empty())
                packedData.insert(packedData.end(), w3.begin(), w3.end());

            offsets.b3Offset = static_cast<uint32_t>(packedData.size());
            if (!b3.empty())
                packedData.insert(packedData.end(), b3.begin(), b3.end());

            struct LoadedDecoder
            {
                int32_t mlpWidth = 0;
                int32_t mlpDepth = 0;
                ref<Buffer> decoderBuffer;
                Data::DecoderWeightOffsets offsets;
            };

            LoadedDecoder loaded;
            loaded.mlpWidth = mlpWidth;
            loaded.mlpDepth = mlpDepth;
            loaded.offsets = offsets;
            loaded.decoderBuffer = make_ref<Buffer>(
                mpDevice,
                sizeof(float),
                static_cast<uint32_t>(packedData.size()),
                ResourceBindFlags::ShaderResource,
                MemoryType::DeviceLocal,
                packedData.data(),
                false
            );
            return loaded;
        };

        auto brdf = loadDecoderWeights(weightsPath, 20, 3, true);
        mpBrdfDecoderBuffer = brdf.decoderBuffer;

        mData.brdfMlpWidth = static_cast<uint32_t>(brdf.mlpWidth);
        mData.brdfMlpDepth = static_cast<uint32_t>(brdf.mlpDepth);
        mData.brdfWeightOffsets = brdf.offsets;


        auto sampler = loadDecoderWeights(samplerWeightsPath, 8 + 3, 9, false);
        mpSamplerDecoderBuffer = sampler.decoderBuffer;

        mData.samplerMlpWidth = static_cast<uint32_t>(sampler.mlpWidth);
        mData.samplerMlpDepth = static_cast<uint32_t>(sampler.mlpDepth);
        mData.samplerWeightOffsets = sampler.offsets;

        if (!mpSampler)
        {
            mpSampler = createLatentSampler();
        }

    }

    uint32_t NeuralMaterial::uploadBuffer(MaterialSystem* pOwner, const ref<Buffer>& pBuffer, uint32_t& id)
    {
        FALCOR_ASSERT(pBuffer);
        if (id == uint32_t(-1)) id = pOwner->addBuffer(pBuffer);
        else pOwner->replaceBuffer(id, pBuffer);
        return id;
    }

    Material::UpdateFlags NeuralMaterial::update(MaterialSystem* pOwner)
    {
        UpdateFlags updates = mUpdates;
        mUpdates = UpdateFlags::None;

        if (!mpSampler)
        {
            mpSampler = createLatentSampler();
            updates |= UpdateFlags::ResourcesChanged;
        }

        updateDefaultTextureSamplerID(pOwner, mpSampler);
        updateTextureHandle(pOwner, mpLatent0, mData.texLatent0);
        updateTextureHandle(pOwner, mpLatent1, mData.texLatent1);

        // BRDF Decoder
        uploadBuffer(pOwner, mpBrdfDecoderBuffer, mData.brdfDecoderBufferID);

        // Sampler Decoder
        uploadBuffer(pOwner, mpSamplerDecoderBuffer, mData.samplerDecoderBufferID);

        return updates;
    }

    bool NeuralMaterial::isEqual(const ref<Material>& pOther) const
    {
        auto p = dynamic_ref_cast<NeuralMaterial>(pOther);
        if (!p) return false;
        return isBaseEqual(*p) &&
               mBasePath == p->mBasePath &&
               mpLatent0 == p->mpLatent0 &&
               mpLatent1 == p->mpLatent1 &&
               mpSampler == p->mpSampler &&
               mLatentLodBias == p->mLatentLodBias &&
               mpBrdfDecoderBuffer == p->mpBrdfDecoderBuffer &&
               mpSamplerDecoderBuffer == p->mpSamplerDecoderBuffer;
    }

    MaterialDataBlob NeuralMaterial::getDataBlob() const
    {
        return prepareDataBlob(mData);
    }

    ProgramDesc::ShaderModuleList NeuralMaterial::getShaderModules() const
    {
        return { ProgramDesc::ShaderModule::fromFile(kShaderFile) };
    }

    TypeConformanceList NeuralMaterial::getTypeConformances() const
    {
        TypeConformanceList conformances;
        // This maps the Slang struct "NeuralMaterial" to the interface "IMaterial"
        conformances.add("NeuralMaterial", "IMaterial");
        return conformances;
    }

}
