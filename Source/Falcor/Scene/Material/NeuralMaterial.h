#pragma once

#include "Material.h"
#include "Core/API/Buffer.h"
#include "Core/API/Sampler.h"
#include "Core/API/Texture.h"
#include "Scene/Material/MaterialTypeRegistry.h"
#include <filesystem>

namespace Falcor
{
    /**
     * Minimal neural material for Falcor's material system.
     *
     * Notes:
     * - This version intentionally supports only the eval decoder.
     * - No latent MIP levels are used; the shader samples mip 0 explicitly.
     * - Importance sampling falls back to cosine hemisphere sampling.
     * - It reuses MaterialType::RGL as a temporary material type slot. If your branch already uses RGL,
     *   add a dedicated MaterialType::Neural in your registry/enum files and replace the type below.
     */
    class NeuralMaterial : public Material
    {
        FALCOR_OBJECT(NeuralMaterial)

    public:
        static ref<NeuralMaterial> create(ref<Device> pDevice, const std::string& name, const std::filesystem::path& basePath)
        {
            return make_ref<NeuralMaterial>(std::move(pDevice), name, basePath);
        }

        NeuralMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& basePath);

        const std::filesystem::path& getBasePath() const { return mBasePath; }
        void setBasePath(const std::filesystem::path& path);

        bool renderUI(Gui::Widgets& widget) override;
        UpdateFlags update(MaterialSystem* pOwner) override;
        bool isEqual(const ref<Material>& pOther) const override;
        MaterialDataBlob getDataBlob() const override;
        ProgramDesc::ShaderModuleList getShaderModules() const override;
        TypeConformanceList getTypeConformances() const override;

        size_t getMaxTextureCount() const override { return 2; }
        size_t getMaxBufferCount() const override { return 7; }
        size_t getMaterialInstanceByteSize() const override { return 128; }

        void setDefaultTextureSampler(const ref<Sampler>& pSampler) override { mpSampler = pSampler; }
        ref<Sampler> getDefaultTextureSampler() const override { return mpSampler; }

    private:
        struct Data
        {
            TextureHandle texLatent0;
            TextureHandle texLatent1;

            uint32_t frameLinearBufferID = uint32_t(-1);
            uint32_t W0BufferID = uint32_t(-1);
            uint32_t B0BufferID = uint32_t(-1);
            uint32_t W1BufferID = uint32_t(-1);
            uint32_t B1BufferID = uint32_t(-1);
            uint32_t W2BufferID = uint32_t(-1);
            uint32_t B2BufferID = uint32_t(-1);

            uint32_t applyExp = 1;
            float expOffset = 3.f;
            uint32_t _pad0 = 0;
            uint32_t _pad1 = 0;
        };
        static_assert(sizeof(Data) <= sizeof(MaterialPayload), "NeuralMaterial payload must fit in MaterialPayload");

        void loadAssets();
        static std::vector<float> readFloatArray(std::ifstream& f, size_t count);
        uint32_t uploadBuffer(MaterialSystem* pOwner, const ref<Buffer>& pBuffer, uint32_t& id);

        std::filesystem::path mBasePath;
        ref<Texture> mpLatent0;
        ref<Texture> mpLatent1;
        ref<Sampler> mpSampler;

        ref<Buffer> mpFrameLinear;
        ref<Buffer> mpW0;
        ref<Buffer> mpB0;
        ref<Buffer> mpW1;
        ref<Buffer> mpB1;
        ref<Buffer> mpW2;
        ref<Buffer> mpB2;

        Data mData = {};
    };
}
