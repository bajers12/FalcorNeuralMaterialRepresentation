import Scene.Scene;
import Scene.Material.MaterialSystem;
import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;

static const float kPi = 3.14159265358979323846f;
static const float kInvPi = 0.31830988618379067154f;

struct NeuralMaterialData
{
    TextureHandle texLatent0 = {};
    TextureHandle texLatent1 = {};

    uint frameLinearBufferID = 0;
    uint W0BufferID = 0;
    uint B0BufferID = 0;
    uint W1BufferID = 0;
    uint B1BufferID = 0;
    uint W2BufferID = 0;
    uint B2BufferID = 0;

    uint applyExp = 1;
    float expOffset = 3.f;
    uint _pad0 = 0;
    uint _pad1 = 0;
};

float neuralBufferLoad(ByteAddressBuffer buf, uint index)
{
    return asfloat(buf.Load(index * 4));
}

void fetchNeuralLatents(const MaterialSystem materials, const NeuralMaterialData md, const uint materialID, const float2 uv, out float z[8])
{
    SamplerState s = materials.getDefaultTextureSampler(materialID);
    float4 z0 = materials.sampleTexture(md.texLatent0, s, uv, 0.f);
    float4 z1 = materials.sampleTexture(md.texLatent1, s, uv, 0.f);

    z[0] = z0.x; z[1] = z0.y; z[2] = z0.z; z[3] = z0.w;
    z[4] = z1.x; z[5] = z1.y; z[6] = z1.z; z[7] = z1.w;
}

float3 evalNeuralBRDFFromLatents(const MaterialSystem materials, const NeuralMaterialData md, const float z[8], float3 wiLocal, float3 woLocal)
{
    // Match the current training/data-generation convention:
    // both wi and wo are local directions pointing away from the surface.
    if (wiLocal.z <= 0.f || woLocal.z <= 0.f)
        return 0.f.xxx;

    ByteAddressBuffer frameLinear = materials.getBuffer(md.frameLinearBufferID);
    ByteAddressBuffer W0 = materials.getBuffer(md.W0BufferID);
    ByteAddressBuffer B0 = materials.getBuffer(md.B0BufferID);
    ByteAddressBuffer W1 = materials.getBuffer(md.W1BufferID);
    ByteAddressBuffer B1 = materials.getBuffer(md.B1BufferID);
    ByteAddressBuffer W2 = materials.getBuffer(md.W2BufferID);
    ByteAddressBuffer B2 = materials.getBuffer(md.B2BufferID);

    float sf[12];
    [unroll]
    for (uint r = 0; r < 12; ++r)
    {
        float s = 0.f;
        [unroll]
        for (uint c = 0; c < 8; ++c)
            s += neuralBufferLoad(frameLinear, r * 8 + c) * z[c];
        sf[r] = s;
    }

    float x[20];

    // Trainer contract:
    //   - frame_linear(z) -> (Nxyz, Txyz) per frame, no +1 offsets
    //   - B = normalize(cross(N, T))
    //   - MLP input is [z, dir_feats]
    //   - dir_feats flatten as [frame0(wi_TBN, wo_TBN), frame1(...)]
    [unroll]
    for (uint i = 0; i < 8; ++i) x[i] = z[i];

    [unroll]
    for (uint f = 0; f < 2; ++f)
    {
        uint sfBase = 6 * f;
        uint xBase = 8 + 6 * f;

        float3 N = normalize(float3(sf[sfBase + 0], sf[sfBase + 1], sf[sfBase + 2]));
        float3 T = normalize(float3(sf[sfBase + 3], sf[sfBase + 4], sf[sfBase + 5]));
        float3 B = normalize(cross(N, T));

        float3 wiPred = float3(dot(wiLocal, T), dot(wiLocal, B), dot(wiLocal, N));
        float3 woPred = float3(dot(woLocal, T), dot(woLocal, B), dot(woLocal, N));

        x[xBase + 0] = wiPred.x;
        x[xBase + 1] = wiPred.y;
        x[xBase + 2] = wiPred.z;
        x[xBase + 3] = woPred.x;
        x[xBase + 4] = woPred.y;
        x[xBase + 5] = woPred.z;
    }

    float h0[32];
    [unroll]
    for (uint r = 0; r < 32; ++r)
    {
        float s = neuralBufferLoad(B0, r);
        [unroll]
        for (uint c = 0; c < 20; ++c)
            s += neuralBufferLoad(W0, r * 20 + c) * x[c];
        h0[r] = max(s, 0.f);
    }

    float h1[32];
    [unroll]
    for (uint r = 0; r < 32; ++r)
    {
        float s = neuralBufferLoad(B1, r);
        [unroll]
        for (uint c = 0; c < 32; ++c)
            s += neuralBufferLoad(W1, r * 32 + c) * h0[c];
        h1[r] = max(s, 0.f);
    }

    float3 y = 0.f.xxx;
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = neuralBufferLoad(B2, r);
        [unroll]
        for (uint c = 0; c < 32; ++c)
            s += neuralBufferLoad(W2, r * 32 + c) * h1[c];

        if (r == 0) y.x = s;
        else if (r == 1) y.y = s;
        else y.z = s;
    }

    float3 outValue = md.applyExp != 0 ? exp(y - md.expOffset) : y;
    if (any(isnan(outValue)) || any(isinf(outValue))) return 0.f.xxx;
    return max(outValue, 0.f.xxx);
}

float evalNeuralPdf(const float3 woLocal)
{
    return woLocal.z > 0.f ? woLocal.z * kInvPi : 0.f;
}

float3 sampleCosineHemisphere(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.f * kPi * u.y;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.f, 1.f - u.x));
    return float3(x, y, z);
}

__exported import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Scene.Material.MaterialSystem;

struct NeuralMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    NeuralMaterialData data = {};
    uint defaultSamplerID = 0;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float z[8];
        SamplerState s = gScene.materials.getTextureSampler(defaultSamplerID);
        float4 z0 = gScene.materials.sampleTexture(data.texLatent0, s, sd.uv, 0.f);
        float4 z1 = gScene.materials.sampleTexture(data.texLatent1, s, sd.uv, 0.f);

        z[0] = z0.x; z[1] = z0.y; z[2] = z0.z; z[3] = z0.w;
        z[4] = z1.x; z[5] = z1.y; z[6] = z1.z; z[7] = z1.w;

        float3 wiLocal = sd.frame.toLocal(sd.V);
        float3 woLocal = sd.frame.toLocal(wo);
        return evalNeuralBRDFFromLatents(gScene.materials, data, z, wiLocal, woLocal);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return no_diff eval(sd, wo, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        result = (BSDFSample)0;

        float3 woLocal = sampleCosineHemisphere(sampleNext2D(sg));
        float pdf = evalNeuralPdf(woLocal);
        if (pdf <= 0.f) return false;

        SamplerState s = gScene.materials.getTextureSampler(defaultSamplerID);
        float4 z0 = gScene.materials.sampleTexture(data.texLatent0, s, sd.uv, 0.f);
        float4 z1 = gScene.materials.sampleTexture(data.texLatent1, s, sd.uv, 0.f);

        float z[8];
        z[0] = z0.x; z[1] = z0.y; z[2] = z0.z; z[3] = z0.w;
        z[4] = z1.x; z[5] = z1.y; z[6] = z1.z; z[7] = z1.w;

        float3 wiLocal = sd.frame.toLocal(sd.V);
        float3 fcos = evalNeuralBRDFFromLatents(gScene.materials, data, z, wiLocal, woLocal);

        result.wo = sd.frame.fromLocal(woLocal);
        result.pdf = pdf;
        result.weight = fcos / pdf;
        result.lobeType = uint(LobeType::DiffuseReflection);
        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        return evalNeuralPdf(sd.frame.toLocal(wo));
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.emission = 0.f.xxx;
        p.roughness = 1.f;
        p.guideNormal = sd.frame.N;
        p.flags = 0;
        p.diffuseReflectionAlbedo = 0.5.xxx;
        p.diffuseTransmissionAlbedo = 0.f.xxx;
        p.specularReflectionAlbedo = 0.f.xxx;
        p.specularTransmissionAlbedo = 0.f.xxx;
        p.specularReflectance = 0.04.xxx;
        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return uint(LobeType::DiffuseReflection);
    }
};

struct NeuralMaterial : MaterialBase, IMaterial
{
    NeuralMaterialData data;

    typedef NeuralMaterialInstance MaterialInstance;

    NeuralMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        NeuralMaterialInstance m = {};
        m.data = data;
        m.defaultSamplerID = header.getDefaultTextureSamplerID();
        return m;
    }

    [Differentiable]
    NeuralMaterialInstance setupDiffMaterialInstance( out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints )
    {
        diffData = {};
        return setupMaterialInstance(ms, sd, lod, hints);

    }
};