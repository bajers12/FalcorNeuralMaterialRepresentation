import Scene.Scene;
import Scene.Material.MaterialSystem;
import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Utils.Math.ShadingFrame;

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
    uint W3BufferID = 0;
    uint B3BufferID = 0;

    uint applyExp = 1;
    float expOffset = 3.f;
    uint mlpWidth = 32;
    uint mlpDepth = 2;
};

float neuralBufferLoad(ByteAddressBuffer buf, uint index)
{
    return asfloat(buf.Load(index * 4));
}

float4 neuralBufferLoad4(ByteAddressBuffer buf, uint index)
{
    return asfloat(buf.Load4(index * 4));
}

float3 neuralSafeNormalize(float3 v)
{
    return v * rsqrt(max(dot(v, v), 1e-16f));
}

float3 evalMLP16x2(ByteAddressBuffer W0, ByteAddressBuffer B0, ByteAddressBuffer W1, ByteAddressBuffer B1, ByteAddressBuffer W2, ByteAddressBuffer B2, const float x[20])
{
    float4 x4[5];
    [unroll]
    for (uint i = 0; i < 5; ++i)
    {
        uint base = 4 * i;
        x4[i] = float4(x[base + 0], x[base + 1], x[base + 2], x[base + 3]);
    }

    float h0[16];
    [unroll]
    for (uint r = 0; r < 16; ++r)
    {
        float s = neuralBufferLoad(B0, r);
        [unroll]
        for (uint g = 0; g < 5; ++g)
            s += dot(neuralBufferLoad4(W0, r * 20 + g * 4), x4[g]);
        h0[r] = max(s, 0.f);
    }

    float h1[16];
    [unroll]
    for (uint r = 0; r < 16; ++r)
    {
        float s = neuralBufferLoad(B1, r);
        [unroll]
        for (uint g = 0; g < 4; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h0[base + 0], h0[base + 1], h0[base + 2], h0[base + 3]);
            s += dot(neuralBufferLoad4(W1, r * 16 + base), h);
        }
        h1[r] = max(s, 0.f);
    }

    float3 y = 0.f.xxx;
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = neuralBufferLoad(B2, r);
        [unroll]
        for (uint g = 0; g < 4; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h1[base + 0], h1[base + 1], h1[base + 2], h1[base + 3]);
            s += dot(neuralBufferLoad4(W2, r * 16 + base), h);
        }
        if (r == 0) y.x = s;
        else if (r == 1) y.y = s;
        else y.z = s;
    }
    return y;
}

float3 evalMLP32x2(ByteAddressBuffer W0, ByteAddressBuffer B0, ByteAddressBuffer W1, ByteAddressBuffer B1, ByteAddressBuffer W2, ByteAddressBuffer B2, const float x[20])
{
    float4 x4[5];
    [unroll]
    for (uint i = 0; i < 5; ++i)
    {
        uint base = 4 * i;
        x4[i] = float4(x[base + 0], x[base + 1], x[base + 2], x[base + 3]);
    }

    float h0[32];
    [unroll]
    for (uint r = 0; r < 32; ++r)
    {
        float s = neuralBufferLoad(B0, r);
        [unroll]
        for (uint g = 0; g < 5; ++g)
            s += dot(neuralBufferLoad4(W0, r * 20 + g * 4), x4[g]);
        h0[r] = max(s, 0.f);
    }

    float h1[32];
    [unroll]
    for (uint r = 0; r < 32; ++r)
    {
        float s = neuralBufferLoad(B1, r);
        [unroll]
        for (uint g = 0; g < 8; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h0[base + 0], h0[base + 1], h0[base + 2], h0[base + 3]);
            s += dot(neuralBufferLoad4(W1, r * 32 + base), h);
        }
        h1[r] = max(s, 0.f);
    }

    float3 y = 0.f.xxx;
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = neuralBufferLoad(B2, r);
        [unroll]
        for (uint g = 0; g < 8; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h1[base + 0], h1[base + 1], h1[base + 2], h1[base + 3]);
            s += dot(neuralBufferLoad4(W2, r * 32 + base), h);
        }
        if (r == 0) y.x = s;
        else if (r == 1) y.y = s;
        else y.z = s;
    }
    return y;
}

float3 evalMLP64x2(ByteAddressBuffer W0, ByteAddressBuffer B0, ByteAddressBuffer W1, ByteAddressBuffer B1, ByteAddressBuffer W2, ByteAddressBuffer B2, const float x[20])
{
    float4 x4[5];
    [unroll]
    for (uint i = 0; i < 5; ++i)
    {
        uint base = 4 * i;
        x4[i] = float4(x[base + 0], x[base + 1], x[base + 2], x[base + 3]);
    }

    float h0[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoad(B0, r);
        [unroll]
        for (uint g = 0; g < 5; ++g)
            s += dot(neuralBufferLoad4(W0, r * 20 + g * 4), x4[g]);
        h0[r] = max(s, 0.f);
    }

    float h1[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoad(B1, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h0[base + 0], h0[base + 1], h0[base + 2], h0[base + 3]);
            s += dot(neuralBufferLoad4(W1, r * 64 + base), h);
        }
        h1[r] = max(s, 0.f);
    }

    float3 y = 0.f.xxx;
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = neuralBufferLoad(B2, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h1[base + 0], h1[base + 1], h1[base + 2], h1[base + 3]);
            s += dot(neuralBufferLoad4(W2, r * 64 + base), h);
        }
        if (r == 0) y.x = s;
        else if (r == 1) y.y = s;
        else y.z = s;
    }
    return y;
}

float3 evalMLP64x3(ByteAddressBuffer W0, ByteAddressBuffer B0, ByteAddressBuffer W1, ByteAddressBuffer B1, ByteAddressBuffer W2, ByteAddressBuffer B2, ByteAddressBuffer W3, ByteAddressBuffer B3, const float x[20])
{
    float4 x4[5];
    [unroll]
    for (uint i = 0; i < 5; ++i)
    {
        uint base = 4 * i;
        x4[i] = float4(x[base + 0], x[base + 1], x[base + 2], x[base + 3]);
    }

    float h0[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoad(B0, r);
        [unroll]
        for (uint g = 0; g < 5; ++g)
            s += dot(neuralBufferLoad4(W0, r * 20 + g * 4), x4[g]);
        h0[r] = max(s, 0.f);
    }

    float h1[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoad(B1, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h0[base + 0], h0[base + 1], h0[base + 2], h0[base + 3]);
            s += dot(neuralBufferLoad4(W1, r * 64 + base), h);
        }
        h1[r] = max(s, 0.f);
    }

    float h2[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoad(B2, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h1[base + 0], h1[base + 1], h1[base + 2], h1[base + 3]);
            s += dot(neuralBufferLoad4(W2, r * 64 + base), h);
        }
        h2[r] = max(s, 0.f);
    }

    float3 y = 0.f.xxx;
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = neuralBufferLoad(B3, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h2[base + 0], h2[base + 1], h2[base + 2], h2[base + 3]);
            s += dot(neuralBufferLoad4(W3, r * 64 + base), h);
        }
        if (r == 0) y.x = s;
        else if (r == 1) y.y = s;
        else y.z = s;
    }
    return y;
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
    ByteAddressBuffer W3 = materials.getBuffer(md.W3BufferID);
    ByteAddressBuffer B3 = materials.getBuffer(md.B3BufferID);

    if (md.mlpWidth == 0 || md.mlpWidth > 64)
        return 0.f.xxx;
    if (md.mlpDepth != 2 && md.mlpDepth != 3)
        return 0.f.xxx;

    float4 z4a = float4(z[0], z[1], z[2], z[3]);
    float4 z4b = float4(z[4], z[5], z[6], z[7]);

    float sf[12];
    [unroll]
    for (uint r = 0; r < 12; ++r)
    {
        float s = 0.f;
        s += dot(neuralBufferLoad4(frameLinear, r * 8 + 0), z4a);
        s += dot(neuralBufferLoad4(frameLinear, r * 8 + 4), z4b);
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

        float3 N = neuralSafeNormalize(float3(sf[sfBase + 0], sf[sfBase + 1], sf[sfBase + 2]));
        float3 Traw = float3(sf[sfBase + 3], sf[sfBase + 4], sf[sfBase + 5]);
        float3 T = neuralSafeNormalize(Traw - dot(Traw, N) * N);
        float3 B = cross(N, T);

        float3 wiPred = float3(dot(wiLocal, T), dot(wiLocal, B), dot(wiLocal, N));
        float3 woPred = float3(dot(woLocal, T), dot(woLocal, B), dot(woLocal, N));

        x[xBase + 0] = wiPred.x;
        x[xBase + 1] = wiPred.y;
        x[xBase + 2] = wiPred.z;
        x[xBase + 3] = woPred.x;
        x[xBase + 4] = woPred.y;
        x[xBase + 5] = woPred.z;
    }

    float3 y = 0.f.xxx;
    if (md.mlpWidth == 16 && md.mlpDepth == 2)
        y = evalMLP16x2(W0, B0, W1, B1, W2, B2, x);
    else if (md.mlpWidth == 32 && md.mlpDepth == 2)
        y = evalMLP32x2(W0, B0, W1, B1, W2, B2, x);
    else if (md.mlpWidth == 64 && md.mlpDepth == 2)
        y = evalMLP64x2(W0, B0, W1, B1, W2, B2, x);
    else if (md.mlpWidth == 64 && md.mlpDepth == 3)
        y = evalMLP64x3(W0, B0, W1, B1, W2, B2, W3, B3, x);
    else
        return 0.f.xxx;

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
