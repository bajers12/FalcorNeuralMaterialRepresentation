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

            struct LoadedDecoder
            {
                int32_t mlpWidth = 0;
                int32_t mlpDepth = 0;
                ref<Buffer> frameLinear;
                ref<Buffer> w0;
                ref<Buffer> b0;
                ref<Buffer> w1;
                ref<Buffer> b1;
                ref<Buffer> w2;
                ref<Buffer> b2;
                ref<Buffer> w3;
                ref<Buffer> b3;
            };

            LoadedDecoder loaded;
            loaded.mlpWidth = mlpWidth;
            loaded.mlpDepth = mlpDepth;
            loaded.frameLinear = makeStructured(frameLinear);
            loaded.w0 = makeStructured(w0);
            loaded.b0 = makeStructured(b0);
            loaded.w1 = makeStructured(w1);
            loaded.b1 = makeStructured(b1);
            loaded.w2 = makeStructured(w2);
            loaded.b2 = makeStructured(b2);
            loaded.w3 = w3.empty() ? makeStructured(std::vector<float>{ 0.f }) : makeStructured(w3);
            loaded.b3 = b3.empty() ? makeStructured(std::vector<float>{ 0.f }) : makeStructured(b3);
            return loaded;
        };

        auto brdf = loadDecoderWeights(weightsPath, 20, 3);
        mpFrameLinear = brdf.frameLinear;
        mpW0 = brdf.w0;
        mpB0 = brdf.b0;
        mpW1 = brdf.w1;
        mpB1 = brdf.b1;
        mpW2 = brdf.w2;
        mpB2 = brdf.b2;
        mpW3 = brdf.w3;
        mpB3 = brdf.b3;

        mData.mlpWidth = static_cast<uint32_t>(brdf.mlpWidth);
        mData.mlpDepth = static_cast<uint32_t>(brdf.mlpDepth);

        if (std::filesystem::exists(samplerWeightsPath))
        {
            auto sampler = loadDecoderWeights(samplerWeightsPath, 14, 10);

            mpSamplerFrameLinear = sampler.frameLinear;
            mpSamplerW0 = sampler.w0;
            mpSamplerB0 = sampler.b0;
            mpSamplerW1 = sampler.w1;
            mpSamplerB1 = sampler.b1;
            mpSamplerW2 = sampler.w2;
            mpSamplerB2 = sampler.b2;
        }

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

        uploadBuffer(pOwner, mpFrameLinear, mData.frameLinearBufferID);
        uploadBuffer(pOwner, mpW0, mData.W0BufferID);
        uploadBuffer(pOwner, mpB0, mData.B0BufferID);
        uploadBuffer(pOwner, mpW1, mData.W1BufferID);
        uploadBuffer(pOwner, mpB1, mData.B1BufferID);
        uploadBuffer(pOwner, mpW2, mData.W2BufferID);
        uploadBuffer(pOwner, mpB2, mData.B2BufferID);
        uploadBuffer(pOwner, mpW3, mData.W3BufferID);
        uploadBuffer(pOwner, mpB3, mData.B3BufferID);

        if (mpSamplerFrameLinear)
        {
            uploadBuffer(pOwner, mpSamplerFrameLinear, mData.samplerFrameLinearBufferID);
            uploadBuffer(pOwner, mpSamplerW0, mData.samplerW0BufferID);
            uploadBuffer(pOwner, mpSamplerB0, mData.samplerB0BufferID);
            uploadBuffer(pOwner, mpSamplerW1, mData.samplerW1BufferID);
            uploadBuffer(pOwner, mpSamplerB1, mData.samplerB1BufferID);
            uploadBuffer(pOwner, mpSamplerW2, mData.samplerW2BufferID);
            uploadBuffer(pOwner, mpSamplerB2, mData.samplerB2BufferID);
        }

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
               mpFrameLinear == p->mpFrameLinear &&
               mpW0 == p->mpW0 && mpB0 == p->mpB0 &&
               mpW1 == p->mpW1 && mpB1 == p->mpB1 &&
               mpW2 == p->mpW2 && mpB2 == p->mpB2 &&
               mpW3 == p->mpW3 && mpB3 == p->mpB3 &&
               mpSamplerFrameLinear == p->mpSamplerFrameLinear &&
               mpSamplerW0 == p->mpSamplerW0 && mpSamplerB0 == p->mpSamplerB0 &&
               mpSamplerW1 == p->mpSamplerW1 && mpSamplerB1 == p->mpSamplerB1 &&
               mpSamplerW2 == p->mpSamplerW2 && mpSamplerB2 == p->mpSamplerB2;
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
