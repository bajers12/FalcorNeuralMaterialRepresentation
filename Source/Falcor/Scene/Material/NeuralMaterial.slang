import Scene.Scene;
import Scene.Material.MaterialSystem;
import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Utils.Math.ShadingFrame;

static const float kPi = 3.14159265358979323846f;
static const float kInvPi = 0.31830988618379067154f;
static const float expOffset = 3.0f;

struct DecoderWeightOffsets
{
    uint frameLinearOffset;  // in floats
    uint w0Offset;
    uint b0Offset;
    uint w1Offset;
    uint b1Offset;
    uint w2Offset;
    uint b2Offset;
    uint w3Offset;
    uint b3Offset;
};

struct NeuralMaterialData
{
    TextureHandle texLatent0 = {};
    TextureHandle texLatent1 = {};
    uint latentMipControl = 1;

    uint brdfDecoderBufferID = 0;
    uint samplerDecoderBufferID = 0;

    uint brdfMlpWidth = 32;
    uint brdfMlpDepth = 2;
    DecoderWeightOffsets brdfWeightOffsets = {};

    uint samplerMlpWidth = 32;
    uint samplerMlpDepth = 2;
    DecoderWeightOffsets samplerWeightOffsets = {};
};

uint neuralMipCount(const NeuralMaterialData md)
{
    return max(1u, md.latentMipControl & 0xffu);
}

uint neuralForcedMip(const NeuralMaterialData md)
{
    return (md.latentMipControl >> 8) & 0xffu;
}

uint neuralDebugMode(const NeuralMaterialData md)
{
    return (md.latentMipControl >> 16) & 0x3u;
}

uint neuralFilteringMode(const NeuralMaterialData md)
{
    return (md.latentMipControl >> 18) & 0x3u;
}

uint neuralSamplingMode(const NeuralMaterialData md)
{
    return (md.latentMipControl >> 20) & 0x3u;
}

bool neuralForceMipEnabled(const NeuralMaterialData md)
{
    return (md.latentMipControl & (1u << 31)) != 0u;
}

static const uint kNeuralFilteringHardware = 0u;
static const uint kNeuralFilteringPaperStochastic = 1u;
static const uint kNeuralSamplingLearned = 0u;
static const uint kNeuralSamplingCosine = 1u;

bool neuralHasLearnedSampler(const NeuralMaterialData md)
{
    return md.samplerMlpWidth != 0u && (md.samplerMlpDepth == 2u || md.samplerMlpDepth == 3u);
}

struct NeuralMipPair
{
    uint mip0;
    uint mip1;
    float w1;
};

NeuralMipPair neuralMipPair(const NeuralMaterialData md, float lod)
{
    uint count = neuralMipCount(md);
    float clampedLod = clamp(lod, 0.f, float(count - 1u));
    NeuralMipPair p;
    p.mip0 = min(uint(floor(clampedLod)), count - 1u);
    p.mip1 = min(p.mip0 + 1u, count - 1u);
    p.w1 = p.mip0 == p.mip1 ? 0.f : frac(clampedLod);
    return p;
}

float3 neuralMipDebugColor(const NeuralMaterialData md, float lod, const float4 latent0, const float4 latent1)
{
    uint count = neuralMipCount(md);
    uint mip = min(uint(round(lod)), count - 1u);
    float normalizedLod = count > 1u ? float(mip) / float(count - 1u) : 0.f;
    uint mode = neuralDebugMode(md);
    if (mode == 1u) return normalizedLod.xxx;
    if (mode == 2u) return float3(normalizedLod, 1.f - normalizedLod, 0.f);
    if (mode == 3u) return saturate(latent0.xyz * 16.f + 0.5f.xxx);
    return 0.f.xxx;
}

float neuralBufferLoad(ByteAddressBuffer buf, uint index)
{
    return asfloat(buf.Load(index * 4));
}

float4 neuralBufferLoad4(ByteAddressBuffer buf, uint index)
{
    return asfloat(buf.Load4(index * 4));
}

float neuralBufferLoadWithOffset(ByteAddressBuffer buf, uint offset, uint index)
{
    return neuralBufferLoad(buf, offset + index);
}

float4 neuralBufferLoad4WithOffset(ByteAddressBuffer buf, uint offset, uint index)
{
    return neuralBufferLoad4(buf, offset + index);
}

float3 neuralSafeNormalize(float3 v)
{
    return v * rsqrt(max(dot(v, v), 1e-16f));
}

float3 evalMLP16x2(
    ByteAddressBuffer buf,
    const DecoderWeightOffsets offsets,
    const float x[20]
)
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
        float s = neuralBufferLoadWithOffset(buf, offsets.b0Offset, r);
        [unroll]
        for (uint g = 0; g < 5; ++g)
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w0Offset, r * 20 + g * 4), x4[g]);
        h0[r] = max(s, 0.f);
    }

    float h1[16];
    [unroll]
    for (uint r = 0; r < 16; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b1Offset, r);
        [unroll]
        for (uint g = 0; g < 4; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h0[base + 0], h0[base + 1], h0[base + 2], h0[base + 3]);
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w1Offset, r * 16 + base), h);
        }
        h1[r] = max(s, 0.f);
    }

    float3 y = 0.f.xxx;
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b2Offset, r);
        [unroll]
        for (uint g = 0; g < 4; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h1[base + 0], h1[base + 1], h1[base + 2], h1[base + 3]);
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w2Offset, r * 16 + base), h);
        }
        if (r == 0)
            y.x = s;
        else if (r == 1)
            y.y = s;
        else
            y.z = s;
    }
    return y;
}

float3 evalMLP32x2(
    ByteAddressBuffer buf,
    const DecoderWeightOffsets offsets,
    const float x[20]
)
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
        float s = neuralBufferLoadWithOffset(buf, offsets.b0Offset, r);
        [unroll]
        for (uint g = 0; g < 5; ++g)
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w0Offset, r * 20 + g * 4), x4[g]);
        h0[r] = max(s, 0.f);
    }

    float h1[32];
    [unroll]
    for (uint r = 0; r < 32; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b1Offset, r);
        [unroll]
        for (uint g = 0; g < 8; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h0[base + 0], h0[base + 1], h0[base + 2], h0[base + 3]);
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w1Offset, r * 32 + base), h);
        }
        h1[r] = max(s, 0.f);
    }

    float3 y = 0.f.xxx;
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b2Offset, r);
        [unroll]
        for (uint g = 0; g < 8; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h1[base + 0], h1[base + 1], h1[base + 2], h1[base + 3]);
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w2Offset, r * 32 + base), h);
        }
        if (r == 0)
            y.x = s;
        else if (r == 1)
            y.y = s;
        else
            y.z = s;
    }
    return y;
}

float3 evalMLP64x2(
    ByteAddressBuffer buf,
    const DecoderWeightOffsets offsets,
    const float x[20]
)
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
        float s = neuralBufferLoadWithOffset(buf, offsets.b0Offset, r);
        [unroll]
        for (uint g = 0; g < 5; ++g)
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w0Offset, r * 20 + g * 4), x4[g]);
        h0[r] = max(s, 0.f);
    }

    float h1[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b1Offset, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h0[base + 0], h0[base + 1], h0[base + 2], h0[base + 3]);
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w1Offset, r * 64 + base), h);
        }
        h1[r] = max(s, 0.f);
    }

    float3 y = 0.f.xxx;
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b2Offset, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h1[base + 0], h1[base + 1], h1[base + 2], h1[base + 3]);
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w2Offset, r * 64 + base), h);
        }
        if (r == 0)
            y.x = s;
        else if (r == 1)
            y.y = s;
        else
            y.z = s;
    }
    return y;
}

float3 evalMLP64x3(
    ByteAddressBuffer buf,
    const DecoderWeightOffsets offsets,
    const float x[20]
)
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
        float s = neuralBufferLoadWithOffset(buf, offsets.b0Offset, r);
        [unroll]
        for (uint g = 0; g < 5; ++g)
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w0Offset, r * 20 + g * 4), x4[g]);
        h0[r] = max(s, 0.f);
    }

    float h1[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b1Offset, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h0[base + 0], h0[base + 1], h0[base + 2], h0[base + 3]);
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w1Offset, r * 64 + base), h);
        }
        h1[r] = max(s, 0.f);
    }

    float h2[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b2Offset, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h1[base + 0], h1[base + 1], h1[base + 2], h1[base + 3]);
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w2Offset, r * 64 + base), h);
        }
        h2[r] = max(s, 0.f);
    }

    float3 y = 0.f.xxx;
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b3Offset, r);
        [unroll]
        for (uint g = 0; g < 16; ++g)
        {
            uint base = 4 * g;
            float4 h = float4(h2[base + 0], h2[base + 1], h2[base + 2], h2[base + 3]);
            s += dot(neuralBufferLoad4WithOffset(buf, offsets.w3Offset, r * 64 + base), h);
        }
        if (r == 0)
            y.x = s;
        else if (r == 1)
            y.y = s;
        else
            y.z = s;
    }
    return y;
}

void fetchNeuralLatentPairAtMip(
    const MaterialSystem materials,
    const NeuralMaterialData md,
    const uint samplerID,
    const float2 uv,
    const uint mipLevel,
    out float4 z0,
    out float4 z1
)
{
    SamplerState s = materials.getTextureSampler(samplerID);
    ExplicitLodTextureSampler lod = { float(mipLevel) };
    z0 = materials.sampleTexture(md.texLatent0, s, uv, 0.f.xxxx, lod);
    z1 = materials.sampleTexture(md.texLatent1, s, uv, 0.f.xxxx, lod);
}

void fetchNeuralLatents(const MaterialSystem materials, const NeuralMaterialData md, const uint samplerID, const float2 uv, const uint mipLevel, out float z[8])
{
    float4 z0;
    float4 z1;
    fetchNeuralLatentPairAtMip(materials, md, samplerID, uv, mipLevel, z0, z1);
    z[0] = z0.x;
    z[1] = z0.y;
    z[2] = z0.z;
    z[3] = z0.w;
    z[4] = z1.x;
    z[5] = z1.y;
    z[6] = z1.z;
    z[7] = z1.w;
}

void unpackNeuralLatents(const float4 z0, const float4 z1, out float z[8])
{
    z[0] = z0.x;
    z[1] = z0.y;
    z[2] = z0.z;
    z[3] = z0.w;
    z[4] = z1.x;
    z[5] = z1.y;
    z[6] = z1.z;
    z[7] = z1.w;
}

float computeNeuralLatentLod(
    const MaterialSystem materials,
    const NeuralMaterialData md,
    const ShadingData sd
)
{
    float value = sd.textureLodValue;
    if (sd.textureLodMode == kTextureLodNeedsTextureDims)
    {
        float2 resolvedUv = sd.uv;
        TextureHandle handle = materials.getResolvedTextureHandle(md.texLatent0, resolvedUv);
        if (handle.getMode() != TextureHandle::Mode::Texture)
            return 0.f;

        uint txw, txh;
        materials.materialTextures[handle.getTextureID()].GetDimensions(txw, txh);
        value = addTextureDimsToFootprintLod(value, txw, txh);
    }
    return clamp(value, 0.f, float(neuralMipCount(md) - 1u));
}

uint chooseNeuralMipStochastic(const NeuralMaterialData md, float lod, float u)
{
    NeuralMipPair p = neuralMipPair(md, lod);
    return (u < p.w1) ? p.mip1 : p.mip0;
}

float neuralLatentDistance(const float4 a0, const float4 a1, const float4 b0, const float4 b1)
{
    float4 d0 = a0 - b0;
    float4 d1 = a1 - b1;
    return dot(d0, d0) + dot(d1, d1);
}

float estimateNearestNeuralMip(
    const MaterialSystem ms,
    const NeuralMaterialData data,
    const SamplerState s,
    const float2 uv,
    const float4 autoLatent0,
    const float4 autoLatent1
)
{
    uint count = min(neuralMipCount(data), 16u);
    float bestMip = 0.f;
    float bestDistance = 3.402823e38f;

    for (uint mip = 0u; mip < count; ++mip)
    {
        ExplicitLodTextureSampler explicitLod = { float(mip) };
        float4 z0 = ms.sampleTexture(data.texLatent0, s, uv, 0.f.xxxx, explicitLod);
        float4 z1 = ms.sampleTexture(data.texLatent1, s, uv, 0.f.xxxx, explicitLod);
        float distance = neuralLatentDistance(autoLatent0, autoLatent1, z0, z1);
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestMip = float(mip);
        }
    }

    return bestMip;
}

float3 evalNeuralBRDFFromLatents(
    const MaterialSystem materials,
    const NeuralMaterialData md,
    const float z[8],
    float3 wiLocal,
    float3 woLocal
)
{
    // Match the current training/data-generation convention:
    // both wi and wo are local directions pointing away from the surface.
    if (wiLocal.z <= 0.f || woLocal.z <= 0.f)
        return 0.f.xxx;

    ByteAddressBuffer decoderBuf = materials.getBuffer(md.brdfDecoderBufferID);

    if (md.brdfMlpWidth == 0 || md.brdfMlpWidth > 64)
        return 0.f.xxx;
    if (md.brdfMlpDepth != 2 && md.brdfMlpDepth != 3)
        return 0.f.xxx;

    float4 z4a = float4(z[0], z[1], z[2], z[3]);
    float4 z4b = float4(z[4], z[5], z[6], z[7]);

    float sf[12];
    [unroll]
    for (uint r = 0; r < 12; ++r)
    {
        float s = 0.f;
        s += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, r * 8 + 0), z4a);
        s += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, r * 8 + 4), z4b);
        sf[r] = s;
    }

    float x[20];

    // Trainer contract:
    //   - frame_linear(z) -> (Nxyz, Txyz) per frame, no +1 offsets
    //   - B = normalize(cross(N, T))
    //   - MLP input is [z, dir_feats]
    //   - dir_feats flatten as [frame0(wi_TBN, wo_TBN), frame1(...)]
    [unroll]
    for (uint i = 0; i < 8; ++i)
        x[i] = z[i];

    [unroll]
    for (uint f = 0; f < 2; ++f)
    {
        uint sfBase = 6 * f;
        uint xBase = 8 + 6 * f;

        // Match the trainer and paper supplemental. The frame extractor predicts
        // residuals around N=(0,0,1), T=(1,0,0), and T is not orthogonalized.
        float3 N = neuralSafeNormalize(float3(sf[sfBase + 0], sf[sfBase + 1], sf[sfBase + 2] + 1.f));
        float3 T = neuralSafeNormalize(float3(sf[sfBase + 3] + 1.f, sf[sfBase + 4], sf[sfBase + 5]));
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
    if (md.brdfMlpWidth == 16 && md.brdfMlpDepth == 2)
        y = evalMLP16x2(decoderBuf, md.brdfWeightOffsets, x);
    else if (md.brdfMlpWidth == 32 && md.brdfMlpDepth == 2)
        y = evalMLP32x2(decoderBuf, md.brdfWeightOffsets, x);
    else if (md.brdfMlpWidth == 64 && md.brdfMlpDepth == 2)
        y = evalMLP64x2(decoderBuf, md.brdfWeightOffsets, x);
    else if (md.brdfMlpWidth == 64 && md.brdfMlpDepth == 3)
        y = evalMLP64x3(decoderBuf, md.brdfWeightOffsets, x);
    else
        return 0.f.xxx;

    float3 outValue = exp(y - expOffset);
    if (any(isnan(outValue)) || any(isinf(outValue)))
        return 0.f.xxx;
    return max(outValue, 0.f.xxx);
}

float3 evalNeuralBRDFAtMip(
    const MaterialSystem materials,
    const NeuralMaterialData md,
    const uint samplerID,
    const float2 uv,
    const uint mipLevel,
    float3 wiLocal,
    float3 woLocal
)
{
    float z[8];
    fetchNeuralLatents(materials, md, samplerID, uv, mipLevel, z);
    return evalNeuralBRDFFromLatents(materials, md, z, wiLocal, woLocal);
}

float3 evalNeuralBRDFMipMixture(
    const MaterialSystem materials,
    const NeuralMaterialData md,
    const uint samplerID,
    const float2 uv,
    const float lod,
    float3 wiLocal,
    float3 woLocal
)
{
    NeuralMipPair p = neuralMipPair(md, lod);
    float3 f0 = evalNeuralBRDFAtMip(materials, md, samplerID, uv, p.mip0, wiLocal, woLocal);
    if (p.w1 <= 0.f) return f0;
    float3 f1 = evalNeuralBRDFAtMip(materials, md, samplerID, uv, p.mip1, wiLocal, woLocal);
    return lerp(f0, f1, p.w1);
}

float evalCosinePdf(const float3 woLocal)
{
    return woLocal.z > 0.f ? woLocal.z * kInvPi : 0.f;
}

struct NeuralSamplerParams
{
    float wDiff;
    float slopeDiffX;
    float slopeDiffY;
    float wSpec;
    float alphaX;
    float alphaY;
    float rho;
    float slopeSpecX;
    float slopeSpecY;
    uint valid;
};

float tanhApprox(float x)
{
    return x / sqrt(1.f + x * x);
}

float sinhApprox(float x)
{
    return x * sqrt(1.f + x * x);
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

void buildBasisFromNormal(const float3 n, out float3 t, out float3 b)
{
    float3 up = abs(n.z) > 0.999f ? float3(1.f, 0.f, 0.f) : float3(0.f, 0.f, 1.f);
    t = neuralSafeNormalize(cross(up, n));
    b = neuralSafeNormalize(cross(n, t));
}

void evalSamplerMLPWithDepth(
    ByteAddressBuffer buf,
    const DecoderWeightOffsets offsets,
    uint width,
    uint depth,
    const float x[11],
    out float y[9]
)
{
    [unroll]
    for (uint i = 0; i < 9; ++i)
        y[i] = 0.f;

    if (width == 0 || width > 64)
        return;
    if (depth != 2 && depth != 3)
        return;

    float h0[64];
    [loop]
    for (uint r = 0; r < width; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b0Offset, r);
        [unroll]
        for (uint c = 0; c < 11; ++c)
            s += neuralBufferLoadWithOffset(buf, offsets.w0Offset, r * 11 + c) * x[c];
        h0[r] = max(s, 0.f);
    }

    float h1[64];
    [loop]
    for (uint r = 0; r < width; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b1Offset, r);
        [loop]
        for (uint c = 0; c < width; ++c)
            s += neuralBufferLoadWithOffset(buf, offsets.w1Offset, r * width + c) * h0[c];
        h1[r] = max(s, 0.f);
    }

    if (depth == 2)
    {
        [unroll]
        for (uint r = 0; r < 9; ++r)
        {
            float s = neuralBufferLoadWithOffset(buf, offsets.b2Offset, r);
            [loop]
            for (uint c = 0; c < width; ++c)
                s += neuralBufferLoadWithOffset(buf, offsets.w2Offset, r * width + c) * h1[c];
            y[r] = s;
        }
        return;
    }

    float h2[64];
    [loop]
    for (uint r = 0; r < width; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b2Offset, r);
        [loop]
        for (uint c = 0; c < width; ++c)
            s += neuralBufferLoadWithOffset(buf, offsets.w2Offset, r * width + c) * h1[c];
        h2[r] = max(s, 0.f);
    }

    [unroll]
    for (uint r = 0; r < 9; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b3Offset, r);
        [loop]
        for (uint c = 0; c < width; ++c)
            s += neuralBufferLoadWithOffset(buf, offsets.w3Offset, r * width + c) * h2[c];
        y[r] = s;
    }
}

NeuralSamplerParams evalNeuralSamplerParams(const MaterialSystem materials, const NeuralMaterialData md, const float z[8], float3 wiLocal)
{
    NeuralSamplerParams p = (NeuralSamplerParams)0;
    p.wDiff = 0.5f;
    p.wSpec = 0.5f;
    p.alphaX = 0.5f;
    p.alphaY = 0.5f;
    p.valid = 0;

    if (wiLocal.z <= 0.f)
        return p;

    ByteAddressBuffer decoderBuf = materials.getBuffer(md.samplerDecoderBufferID);

    float x[11];
    [unroll]
    for (uint i = 0; i < 8; ++i)
        x[i] = z[i];
    x[8] = wiLocal.x;
    x[9] = wiLocal.y;
    x[10] = wiLocal.z;

    float raw[9];
    evalSamplerMLPWithDepth(decoderBuf, md.samplerWeightOffsets, md.samplerMlpWidth, md.samplerMlpDepth, x, raw);

    p.alphaX = 1e-4f + 0.5f * (1.f + tanhApprox(raw[0]));
    p.alphaY = 1e-4f + 0.5f * (1.f + tanhApprox(raw[1]));
    p.rho = clamp(tanhApprox(raw[2]), -0.999f, 0.999f);
    p.slopeSpecX = sinhApprox(raw[3]);
    p.slopeSpecY = sinhApprox(raw[4]);
    p.slopeDiffX = sinhApprox(raw[5]);
    p.slopeDiffY = sinhApprox(raw[6]);

    float wSpecRaw = exp(raw[7]);
    float wDiffRaw = exp(raw[8]);
    float invW = 1.f / max(wSpecRaw + wDiffRaw, 1e-8f);
    p.wSpec = wSpecRaw * invW;
    p.wDiff = wDiffRaw * invW;
    p.valid = 1;
    return p;
}

float evalNeuralDiffusePdfFromParams(const NeuralSamplerParams p, const float3 woLocal)
{
    float3 nd = neuralSafeNormalize(float3(-p.slopeDiffX, -p.slopeDiffY, 1.f));
    return max(dot(woLocal, nd), 0.f) * kInvPi;
}

float evalNeuralSpecularPdfFromParams(const NeuralSamplerParams p, const float3 wiLocal, const float3 woLocal)
{
    if (wiLocal.z <= 0.f || woLocal.z <= 0.f)
        return 0.f;

    float3 wh = neuralSafeNormalize(wiLocal + woLocal);
    wh *= (wh.z >= 0.f) ? 1.f : -1.f;

    float cosTheta = wh.z;
    if (cosTheta <= 1e-4f)
        return 0.f;

    float sx = -wh.x / cosTheta;
    float sy = -wh.y / cosTheta;
    sx -= p.slopeSpecX;
    sy -= p.slopeSpecY;

    float sxStd = sx / max(p.alphaX, 1e-6f);
    float sqrtOneMinusRho = sqrt(max(1.f - p.rho * p.rho, 1e-8f));
    float normalization = 1.f / max(p.alphaX * p.alphaY * sqrtOneMinusRho, 1e-8f);
    float syStd = (p.alphaX * sy - p.rho * p.alphaY * sx) * normalization;

    float r2 = sxStd * sxStd + syStd * syStd;
    float p22Std = 1.f / (kPi * (1.f + r2) * (1.f + r2));
    float p22 = p22Std * normalization;
    float pdfH = p22 / max(cosTheta * cosTheta * cosTheta, 1e-8f);
    float Jh = 4.f * abs(dot(wiLocal, wh));
    return pdfH / max(Jh, 1e-8f);
}

float evalNeuralPdfFromParams(const NeuralMaterialData md, const NeuralSamplerParams p, const float3 wiLocal, const float3 woLocal)
{
    if (woLocal.z <= 0.f)
        return 0.f;
    if (p.valid == 0)
        return evalCosinePdf(woLocal);

    float pd = evalNeuralDiffusePdfFromParams(p, woLocal);
    float ps = evalNeuralSpecularPdfFromParams(p, wiLocal, woLocal);

    float pdf = p.wDiff * pd + p.wSpec * ps;
    return max(pdf, 1e-10f);
}

float evalNeuralPdf(
    const MaterialSystem materials,
    const NeuralMaterialData md,
    const float z[8],
    const float3 wiLocal,
    const float3 woLocal
)
{
    NeuralSamplerParams p = evalNeuralSamplerParams(materials, md, z, wiLocal);
    return evalNeuralPdfFromParams(md, p, wiLocal, woLocal);
}

float evalNeuralPdfAtMip(
    const MaterialSystem materials,
    const NeuralMaterialData md,
    const uint samplerID,
    const float2 uv,
    const uint mipLevel,
    const float3 wiLocal,
    const float3 woLocal
)
{
    float z[8];
    fetchNeuralLatents(materials, md, samplerID, uv, mipLevel, z);
    return evalNeuralPdf(materials, md, z, wiLocal, woLocal);
}

float evalNeuralPdfMipMixture(
    const MaterialSystem materials,
    const NeuralMaterialData md,
    const uint samplerID,
    const float2 uv,
    const float lod,
    const float3 wiLocal,
    const float3 woLocal
)
{
    NeuralMipPair p = neuralMipPair(md, lod);
    float pdf0 = evalNeuralPdfAtMip(materials, md, samplerID, uv, p.mip0, wiLocal, woLocal);
    if (p.w1 <= 0.f) return pdf0;
    float pdf1 = evalNeuralPdfAtMip(materials, md, samplerID, uv, p.mip1, wiLocal, woLocal);
    return lerp(pdf0, pdf1, p.w1);
}

float3 sampleNeuralDirectionFromParams(const NeuralMaterialData md, const NeuralSamplerParams p, const float3 wiLocal, float2 u)
{
    if (p.valid == 0)
        return sampleCosineHemisphere(u);

    bool chooseSpec = u.x < p.wSpec;
    float selectDenom = chooseSpec ? max(p.wSpec, 1e-6f) : max(p.wDiff, 1e-6f);
    float uSelect = chooseSpec ? (u.x / selectDenom) : ((u.x - p.wSpec) / selectDenom);
    uSelect = clamp(uSelect, 1e-6f, 1.f - 1e-6f);
    float uAux = clamp(frac(u.y * 0.754877666f + 0.569840296f), 1e-6f, 1.f - 1e-6f);

    if (!chooseSpec)
    {
        float3 nd = neuralSafeNormalize(float3(-p.slopeDiffX, -p.slopeDiffY, 1.f));
        float3 td;
        float3 bd;
        buildBasisFromNormal(nd, td, bd);

        float3 woLocalSample = sampleCosineHemisphere(float2(uSelect, uAux));
        float3 wo = neuralSafeNormalize(td * woLocalSample.x + bd * woLocalSample.y + nd * woLocalSample.z);
        return wo;
    }

    float sqrtOneMinusRho = sqrt(max(1.f - p.rho * p.rho, 1e-8f));
    float s = sqrt(uSelect) / max(sqrt(1.f - uSelect), 1e-6f);
    float phi = 2.f * kPi * uAux;
    float sxStd = s * cos(phi);
    float syStd = s * sin(phi);

    float sx = p.alphaX * sxStd;
    float sy = p.alphaY * (p.rho * sxStd + sqrtOneMinusRho * syStd);
    sx += p.slopeSpecX;
    sy += p.slopeSpecY;

    float3 wh = neuralSafeNormalize(float3(-sx, -sy, 1.f));
    return neuralSafeNormalize(2.f * dot(wiLocal, wh) * wh - wiLocal);
}

float3 sampleNeuralDirection(const MaterialSystem materials, const NeuralMaterialData md, const float z[8], const float3 wiLocal, float2 u)
{
    NeuralSamplerParams p = evalNeuralSamplerParams(materials, md, z, wiLocal);
    return sampleNeuralDirectionFromParams(md, p, wiLocal, u);
}

__exported import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Scene.Material.MaterialSystem;

struct NeuralMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    NeuralMaterialData data = {};
    uint defaultSamplerID = 0;
    float2 uv = float2(0.f);
    float neuralLod = 0.f;
    float4 latent0 = float4(0.f);
    float4 latent1 = float4(0.f);
    float debugLod = 0.f;

    bool usePaperStochasticFiltering()
    {
        return !neuralForceMipEnabled(data) && neuralFilteringMode(data) == kNeuralFilteringPaperStochastic;
    }

    void fetchLatentsForShading<S : ISampleGenerator>(inout S sg, out float4 z0, out float4 z1, out float selectedMip)
    {
        if (!usePaperStochasticFiltering())
        {
            z0 = latent0;
            z1 = latent1;
            selectedMip = debugLod;
            return;
        }

        uint mip = chooseNeuralMipStochastic(data, neuralLod, sampleNext1D(sg));
        fetchNeuralLatentPairAtMip(gScene.materials, data, defaultSamplerID, uv, mip, z0, z1);
        selectedMip = float(mip);
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sd.frame.toLocal(sd.V);
        float3 woLocal = sd.frame.toLocal(wo);
        float4 z0;
        float4 z1;
        float selectedMip;
        fetchLatentsForShading(sg, z0, z1, selectedMip);

        if (neuralDebugMode(data) != 0u)
        {
            return neuralMipDebugColor(data, selectedMip, z0, z1) * kInvPi;
        }

        float z[8];
        unpackNeuralLatents(z0, z1, z);
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

        float3 wiLocal = sd.frame.toLocal(sd.V);
        float4 z0;
        float4 z1;
        float selectedMip;
        fetchLatentsForShading(sg, z0, z1, selectedMip);
        float2 u = sampleNext2D(sg);

        if (neuralDebugMode(data) != 0u)
        {
            float3 woLocalDebug = sampleCosineHemisphere(u);
            float debugPdf = evalCosinePdf(woLocalDebug);
            if (debugPdf <= 0.f) return false;
            result.wo = sd.frame.fromLocal(woLocalDebug);
            result.pdf = debugPdf;
            result.weight = neuralMipDebugColor(data, selectedMip, z0, z1);
            result.lobeType = uint(LobeType::DiffuseReflection);
            return true;
        }

        float z[8];
        unpackNeuralLatents(z0, z1, z);

        float3 woLocal;
        float pdf;
        uint sampledLobeType = uint(LobeType::DiffuseReflection);
        bool useLearnedSampler =
            useImportanceSampling &&
            neuralSamplingMode(data) == kNeuralSamplingLearned &&
            neuralHasLearnedSampler(data);
        if (useLearnedSampler)
        {
            NeuralSamplerParams p = evalNeuralSamplerParams(gScene.materials, data, z, wiLocal);
            sampledLobeType = (p.valid != 0u && u.x < p.wSpec) ?
                uint(LobeType::SpecularReflection) :
                uint(LobeType::DiffuseReflection);
            woLocal = sampleNeuralDirectionFromParams(data, p, wiLocal, u);
            pdf = evalNeuralPdfFromParams(data, p, wiLocal, woLocal);
        }
        else
        {
            woLocal = sampleCosineHemisphere(u);
            pdf = evalCosinePdf(woLocal);
        }

        if (pdf <= 0.f)
            return false;

        float3 fcos = evalNeuralBRDFFromLatents(gScene.materials, data, z, wiLocal, woLocal);

        if (usePaperStochasticFiltering() && useLearnedSampler)
        {
            NeuralMipPair p = neuralMipPair(data, neuralLod);
            float zMip0[8];
            fetchNeuralLatents(gScene.materials, data, defaultSamplerID, uv, p.mip0, zMip0);

            float pdf0 = evalNeuralPdf(gScene.materials, data, zMip0, wiLocal, woLocal);
            float3 f0 = evalNeuralBRDFFromLatents(gScene.materials, data, zMip0, wiLocal, woLocal);

            if (p.w1 > 0.f)
            {
                float zMip1[8];
                fetchNeuralLatents(gScene.materials, data, defaultSamplerID, uv, p.mip1, zMip1);
                float pdf1 = evalNeuralPdf(gScene.materials, data, zMip1, wiLocal, woLocal);
                float3 f1 = evalNeuralBRDFFromLatents(gScene.materials, data, zMip1, wiLocal, woLocal);
                pdf = lerp(pdf0, pdf1, p.w1);
                fcos = lerp(f0, f1, p.w1);
            }
            else
            {
                pdf = pdf0;
                fcos = f0;
            }
        }

        if (pdf <= 0.f)
            return false;

        result.wo = sd.frame.fromLocal(woLocal);
        result.pdf = pdf;
        result.weight = fcos / pdf;
        result.lobeType = sampledLobeType;
        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 woLocal = sd.frame.toLocal(wo);
        if (!useImportanceSampling)
            return evalCosinePdf(woLocal);

        float3 wiLocal = sd.frame.toLocal(sd.V);
        if (neuralDebugMode(data) != 0u)
            return evalCosinePdf(woLocal);
        if (neuralSamplingMode(data) == kNeuralSamplingCosine || !neuralHasLearnedSampler(data))
            return evalCosinePdf(woLocal);

        if (usePaperStochasticFiltering())
        {
            NeuralMipPair p = neuralMipPair(data, neuralLod);
            float zMip0[8];
            fetchNeuralLatents(gScene.materials, data, defaultSamplerID, uv, p.mip0, zMip0);
            float pdf0 = evalNeuralPdf(gScene.materials, data, zMip0, wiLocal, woLocal);

            if (p.w1 <= 0.f)
                return pdf0;

            float zMip1[8];
            fetchNeuralLatents(gScene.materials, data, defaultSamplerID, uv, p.mip1, zMip1);
            float pdf1 = evalNeuralPdf(gScene.materials, data, zMip1, wiLocal, woLocal);
            return lerp(pdf0, pdf1, p.w1);
        }

        float z[8];
        unpackNeuralLatents(latent0, latent1, z);
        return evalNeuralPdf(gScene.materials, data, z, wiLocal, woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.emission = 0.f.xxx;
        p.roughness = 0.25f;
        p.guideNormal = sd.frame.N;
        p.flags = 0;
        p.diffuseReflectionAlbedo = 0.5.xxx;
        p.diffuseTransmissionAlbedo = 0.f.xxx;
        p.specularReflectionAlbedo = 0.08.xxx;
        p.specularTransmissionAlbedo = 0.f.xxx;
        p.specularReflectance = 0.08.xxx;
        return p;
    }

    uint getLobeTypes(const ShadingData sd) { return uint(LobeType::DiffuseReflection) | uint(LobeType::SpecularReflection); }
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
        m.uv = sd.uv;
        m.neuralLod = computeNeuralLatentLod(ms, data, sd);
        SamplerState s = ms.getTextureSampler(m.defaultSamplerID);
        if (neuralForceMipEnabled(data))
        {
            uint mip = min(neuralForcedMip(data), neuralMipCount(data) - 1u);
            ExplicitLodTextureSampler forcedLod = { float(mip) };
            m.latent0 = ms.sampleTexture(data.texLatent0, s, sd.uv, 0.f.xxxx, forcedLod);
            m.latent1 = ms.sampleTexture(data.texLatent1, s, sd.uv, 0.f.xxxx, forcedLod);
            m.debugLod = float(mip);
        }
        else if (neuralFilteringMode(data) == kNeuralFilteringPaperStochastic)
        {
            m.debugLod = m.neuralLod;
        }
        else
        {
            m.latent0 = ms.sampleTexture(data.texLatent0, s, sd.uv, 0.f.xxxx, lod);
            m.latent1 = ms.sampleTexture(data.texLatent1, s, sd.uv, 0.f.xxxx, lod);
            uint mode = neuralDebugMode(data);
            m.debugLod = (mode == 1u || mode == 2u)
                ? estimateNearestNeuralMip(ms, data, s, sd.uv, m.latent0, m.latent1)
                : 0.f;
        }
        return m;
    }

    [Differentiable]
    NeuralMaterialInstance setupDiffMaterialInstance(
        out DiffMaterialData diffData,
        const MaterialSystem ms,
        const ShadingData sd,
        const ITextureSampler lod,
        const uint hints
    )
    {
        diffData = {};
        return setupMaterialInstance(ms, sd, lod, hints);
    }
};
