#include "NeuralMaterial.h"
#include "MaterialSystem.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
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
        return dirty;
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

        mpLatent0 = Texture::createFromFile(mpDevice, latent0Path.string(), false, false);
        mpLatent1 = Texture::createFromFile(mpDevice, latent1Path.string(), false, false);

        if (!mpLatent0) FALCOR_THROW("Failed to load latent texture: {}", latent0Path.string());
        if (!mpLatent1) FALCOR_THROW("Failed to load latent texture: {}", latent1Path.string());

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

        auto loadDecoderWeights = [&](const std::filesystem::path& path, uint32_t expectedInputDim, uint32_t expectedOutputDim)
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
            if (numFrames != 2) FALCOR_THROW("Expected numFrames == 2, got {} in {}", numFrames, path.string());
            if (mlpWidth != 16 && mlpWidth != 32 && mlpWidth != 64)
                FALCOR_THROW("Expected mlpWidth in {16, 32, 64}, got {} in {}", mlpWidth, path.string());
            if (mlpDepth != 2 && mlpDepth != 3)
                FALCOR_THROW("Expected mlpDepth in {2, 3}, got {} in {}", mlpDepth, path.string());

            auto frameLinear = readFloatArray(f, 12 * 8);
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

            // Frame linear weights
            offsets.frameLinearOffset = static_cast<uint32_t>(packedData.size());
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

        auto brdf = loadDecoderWeights(weightsPath, 20, 3);
        mpBrdfDecoderBuffer = brdf.decoderBuffer;

        mData.brdfMlpWidth = static_cast<uint32_t>(brdf.mlpWidth);
        mData.brdfMlpDepth = static_cast<uint32_t>(brdf.mlpDepth);
        mData.brdfWeightOffsets = brdf.offsets;


        auto sampler = loadDecoderWeights(samplerWeightsPath, 14, 10);
        mpSamplerDecoderBuffer = sampler.decoderBuffer;

        mData.samplerMlpWidth = static_cast<uint32_t>(sampler.mlpWidth);
        mData.samplerMlpDepth = static_cast<uint32_t>(sampler.mlpDepth);
        mData.samplerWeightOffsets = sampler.offsets;

        if (!mpSampler)
        {
            Sampler::Desc desc;
            desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
            desc.setMaxAnisotropy(8);
            mpSampler = mpDevice->createSampler(desc);
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
            Sampler::Desc desc;
            desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
            desc.setMaxAnisotropy(8);
            mpSampler = mpDevice->createSampler(desc);
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
