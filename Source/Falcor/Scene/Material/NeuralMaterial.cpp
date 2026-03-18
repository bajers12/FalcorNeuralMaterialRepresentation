#include "NeuralMaterial.h"
#include "MaterialSystem.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
#include <fstream>

namespace Falcor
{
    namespace
    {
        constexpr const char kWeightMagic[8] = { 'N','M','D','L','W','T','0','1' };
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

        mpLatent0 = Texture::createFromFile(mpDevice, latent0Path.string(), false, false);
        mpLatent1 = Texture::createFromFile(mpDevice, latent1Path.string(), false, false);

        if (!mpLatent0) FALCOR_THROW("Failed to load latent texture: {}", latent0Path.string());
        if (!mpLatent1) FALCOR_THROW("Failed to load latent texture: {}", latent1Path.string());

        std::ifstream f(weightsPath, std::ios::binary);
        if (!f) FALCOR_THROW("Failed to open weight file: {}", weightsPath.string());

        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, kWeightMagic, 8) != 0)
            FALCOR_THROW("Invalid weight file magic in: {}", weightsPath.string());

        int32_t latentCh = 0;
        int32_t numFrames = 0;
        int32_t applyExp = 0;
        float expOffset = 0.f;

        f.read(reinterpret_cast<char*>(&latentCh), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&numFrames), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&applyExp), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&expOffset), sizeof(float));
        if (!f) FALCOR_THROW("Failed reading weight file header: {}", weightsPath.string());

        // This patch targets the current eval-only 2x32 export.
        if (latentCh != 8) FALCOR_THROW("Expected latentCh == 8, got {} in {}", latentCh, weightsPath.string());
        if (numFrames != 2) FALCOR_THROW("Expected numFrames == 2, got {} in {}", numFrames, weightsPath.string());

        auto frameLinear = readFloatArray(f, 12 * 8);
        auto w0 = readFloatArray(f, 32 * 20);
        auto b0 = readFloatArray(f, 32);
        auto w1 = readFloatArray(f, 32 * 32);
        auto b1 = readFloatArray(f, 32);
        auto w2 = readFloatArray(f, 3 * 32);
        auto b2 = readFloatArray(f, 3);

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

        mpFrameLinear = makeStructured(frameLinear);
        mpW0 = makeStructured(w0);
        mpB0 = makeStructured(b0);
        mpW1 = makeStructured(w1);
        mpB1 = makeStructured(b1);
        mpW2 = makeStructured(w2);
        mpB2 = makeStructured(b2);

        if (!mpSampler)
        {
            Sampler::Desc desc;
            desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
            desc.setMaxAnisotropy(8);
            mpSampler = mpDevice->createSampler(desc);
        }

        mData.applyExp = applyExp != 0 ? 1u : 0u;
        mData.expOffset = expOffset;
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
               mpW2 == p->mpW2 && mpB2 == p->mpB2;
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
