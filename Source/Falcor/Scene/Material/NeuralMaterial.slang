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

    uint brdfDecoderBufferID = 0;
    uint samplerDecoderBufferID = 0;

    uint brdfMlpWidth = 32;
    uint brdfMlpDepth = 2;
    uint brdfNumFrames = 2;
    DecoderWeightOffsets brdfWeightOffsets = {};

    uint samplerMlpWidth = 32;
    uint samplerMlpDepth = 2;
    DecoderWeightOffsets samplerWeightOffsets = {};
};

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
    const uint inputDim,
    const float x[26]
)
{
    float h0[16];
    [unroll]
    for (uint r = 0; r < 16; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b0Offset, r);
        [loop]
        for (uint c = 0; c < inputDim; ++c)
            s += neuralBufferLoadWithOffset(buf, offsets.w0Offset, r * inputDim + c) * x[c];
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
    const uint inputDim,
    const float x[26]
)
{
    float h0[32];
    [unroll]
    for (uint r = 0; r < 32; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b0Offset, r);
        [loop]
        for (uint c = 0; c < inputDim; ++c)
            s += neuralBufferLoadWithOffset(buf, offsets.w0Offset, r * inputDim + c) * x[c];
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
    const uint inputDim,
    const float x[26]
)
{
    float h0[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b0Offset, r);
        [loop]
        for (uint c = 0; c < inputDim; ++c)
            s += neuralBufferLoadWithOffset(buf, offsets.w0Offset, r * inputDim + c) * x[c];
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
    const uint inputDim,
    const float x[26]
)
{
    float h0[64];
    [unroll]
    for (uint r = 0; r < 64; ++r)
    {
        float s = neuralBufferLoadWithOffset(buf, offsets.b0Offset, r);
        [loop]
        for (uint c = 0; c < inputDim; ++c)
            s += neuralBufferLoadWithOffset(buf, offsets.w0Offset, r * inputDim + c) * x[c];
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

void fetchNeuralLatents(const MaterialSystem materials, const NeuralMaterialData md, const uint materialID, const float2 uv, out float z[8])
{
    SamplerState s = materials.getDefaultTextureSampler(materialID);
    float4 z0 = materials.sampleTexture(md.texLatent0, s, uv, 0.f);
    float4 z1 = materials.sampleTexture(md.texLatent1, s, uv, 0.f);

    z[0] = z0.x;
    z[1] = z0.y;
    z[2] = z0.z;
    z[3] = z0.w;
    z[4] = z1.x;
    z[5] = z1.y;
    z[6] = z1.z;
    z[7] = z1.w;
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
    if (md.brdfNumFrames != 2 && md.brdfNumFrames != 3)
        return 0.f.xxx;

    ByteAddressBuffer decoderBuf = materials.getBuffer(md.brdfDecoderBufferID);

    if (md.brdfMlpWidth == 0 || md.brdfMlpWidth > 64)
        return 0.f.xxx;
    if (md.brdfMlpDepth != 2 && md.brdfMlpDepth != 3)
        return 0.f.xxx;

    float4 z4a = float4(z[0], z[1], z[2], z[3]);
    float4 z4b = float4(z[4], z[5], z[6], z[7]);

    float sf[18];
    [unroll]
    for (uint i = 0; i < 18; ++i)
        sf[i] = 0.f;

    float x[26];
    [unroll]
    for (uint i = 0; i < 26; ++i)
        x[i] = 0.f;

    [unroll]
    for (uint i = 0; i < 8; ++i)
        x[i] = z[i];

    [loop]
    for (uint f = 0; f < md.brdfNumFrames; ++f)
    {
        uint sfBase = 6 * f;
        uint xBase = 8 + 6 * f;

        float s0 = 0.f;
        s0 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, sfBase * 8 + 0), z4a);
        s0 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, sfBase * 8 + 4), z4b);
        sf[sfBase + 0] = s0;

        float s1 = 0.f;
        s1 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 1) * 8 + 0), z4a);
        s1 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 1) * 8 + 4), z4b);
        sf[sfBase + 1] = s1;

        float s2 = 0.f;
        s2 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 2) * 8 + 0), z4a);
        s2 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 2) * 8 + 4), z4b);
        sf[sfBase + 2] = s2;

        float s3 = 0.f;
        s3 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 3) * 8 + 0), z4a);
        s3 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 3) * 8 + 4), z4b);
        sf[sfBase + 3] = s3;

        float s4 = 0.f;
        s4 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 4) * 8 + 0), z4a);
        s4 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 4) * 8 + 4), z4b);
        sf[sfBase + 4] = s4;

        float s5 = 0.f;
        s5 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 5) * 8 + 0), z4a);
        s5 += dot(neuralBufferLoad4WithOffset(decoderBuf, md.brdfWeightOffsets.frameLinearOffset, (sfBase + 5) * 8 + 4), z4b);
        sf[sfBase + 5] = s5;

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

    uint inputDim = 8 + 6 * md.brdfNumFrames;
    float3 y = 0.f.xxx;
    if (md.brdfMlpWidth == 16 && md.brdfMlpDepth == 2)
        y = evalMLP16x2(decoderBuf, md.brdfWeightOffsets, inputDim, x);
    else if (md.brdfMlpWidth == 32 && md.brdfMlpDepth == 2)
        y = evalMLP32x2(decoderBuf, md.brdfWeightOffsets, inputDim, x);
    else if (md.brdfMlpWidth == 64 && md.brdfMlpDepth == 2)
        y = evalMLP64x2(decoderBuf, md.brdfWeightOffsets, inputDim, x);
    else if (md.brdfMlpWidth == 64 && md.brdfMlpDepth == 3)
        y = evalMLP64x3(decoderBuf, md.brdfWeightOffsets, inputDim, x);
    else
        return 0.f.xxx;

    float3 outValue = exp(y - expOffset);
    if (any(isnan(outValue)) || any(isinf(outValue)))
        return 0.f.xxx;
    return max(outValue, 0.f.xxx);
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

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float z[8];
        SamplerState s = gScene.materials.getTextureSampler(defaultSamplerID);
        float4 z0 = gScene.materials.sampleTexture(data.texLatent0, s, sd.uv, 0.f);
        float4 z1 = gScene.materials.sampleTexture(data.texLatent1, s, sd.uv, 0.f);

        z[0] = z0.x;
        z[1] = z0.y;
        z[2] = z0.z;
        z[3] = z0.w;
        z[4] = z1.x;
        z[5] = z1.y;
        z[6] = z1.z;
        z[7] = z1.w;

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

        SamplerState s = gScene.materials.getTextureSampler(defaultSamplerID);
        float4 z0 = gScene.materials.sampleTexture(data.texLatent0, s, sd.uv, 0.f);
        float4 z1 = gScene.materials.sampleTexture(data.texLatent1, s, sd.uv, 0.f);

        float z[8];
        z[0] = z0.x;
        z[1] = z0.y;
        z[2] = z0.z;
        z[3] = z0.w;
        z[4] = z1.x;
        z[5] = z1.y;
        z[6] = z1.z;
        z[7] = z1.w;

        float3 wiLocal = sd.frame.toLocal(sd.V);
        float2 u = sampleNext2D(sg);

        float3 woLocal;
        float pdf;
        if (useImportanceSampling)
        {
            NeuralSamplerParams p = evalNeuralSamplerParams(gScene.materials, data, z, wiLocal);
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

        result.wo = sd.frame.fromLocal(woLocal);
        result.pdf = pdf;
        result.weight = fcos / pdf;
        result.lobeType = uint(LobeType::DiffuseReflection);
        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 woLocal = sd.frame.toLocal(wo);
        if (!useImportanceSampling)
            return evalCosinePdf(woLocal);

        SamplerState s = gScene.materials.getTextureSampler(defaultSamplerID);
        float4 z0 = gScene.materials.sampleTexture(data.texLatent0, s, sd.uv, 0.f);
        float4 z1 = gScene.materials.sampleTexture(data.texLatent1, s, sd.uv, 0.f);

        float z[8];
        z[0] = z0.x;
        z[1] = z0.y;
        z[2] = z0.z;
        z[3] = z0.w;
        z[4] = z1.x;
        z[5] = z1.y;
        z[6] = z1.z;
        z[7] = z1.w;

        float3 wiLocal = sd.frame.toLocal(sd.V);
        return evalNeuralPdf(gScene.materials, data, z, wiLocal, woLocal);
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

    uint getLobeTypes(const ShadingData sd) { return uint(LobeType::DiffuseReflection); }
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
