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
        * - No latent MIP levels are used; the shader samples mip 0 explicitly.
        * - Importance sampling uses a dedicated decoder when sampler_weights.bin is available.
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
        size_t getMaxBufferCount() const override { return 16; }
        size_t getMaterialInstanceByteSize() const override { return 128; }

        void setDefaultTextureSampler(const ref<Sampler>& pSampler) override { mpSampler = pSampler; }
        ref<Sampler> getDefaultTextureSampler() const override { return mpSampler; }

    private:
        struct Data
        {
            struct DecoderWeightOffsets
            {
                uint32_t frameLinearOffset = 0;
                uint32_t w0Offset = 0;
                uint32_t b0Offset = 0;
                uint32_t w1Offset = 0;
                uint32_t b1Offset = 0;
                uint32_t w2Offset = 0;
                uint32_t b2Offset = 0;
                uint32_t w3Offset = 0;
                uint32_t b3Offset = 0;
            };

            TextureHandle texLatent0;
            TextureHandle texLatent1;

            uint32_t brdfDecoderBufferID = uint32_t(-1);
            uint32_t samplerDecoderBufferID = uint32_t(-1);

            uint32_t brdfMlpWidth = 32;
            uint32_t brdfMlpDepth = 2;
            DecoderWeightOffsets brdfWeightOffsets = {};

            uint32_t samplerMlpWidth = 32;
            uint32_t samplerMlpDepth = 2;
            DecoderWeightOffsets samplerWeightOffsets = {};
        };
        static_assert(sizeof(Data) <= sizeof(MaterialPayload), "NeuralMaterial payload must fit in MaterialPayload");

        void loadAssets();
        static std::vector<float> readFloatArray(std::ifstream& f, size_t count);
        uint32_t uploadBuffer(MaterialSystem* pOwner, const ref<Buffer>& pBuffer, uint32_t& id);

        std::filesystem::path mBasePath;
        ref<Texture> mpLatent0;
        ref<Texture> mpLatent1;
        ref<Sampler> mpSampler;

        ref<Buffer> mpBrdfDecoderBuffer;
        ref<Buffer> mpSamplerDecoderBuffer;

        Data mData = {};
    };
}
