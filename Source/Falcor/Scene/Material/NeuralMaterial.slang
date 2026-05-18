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

float evalCosinePdf(const float3 woLocal)
{
    return woLocal.z > 0.f ? woLocal.z * kInvPi : 0.f;
}

struct NeuralSamplerParams
{
    float wd;
    float muDx;
    float muDy;
    float ws;
    float ax;
    float ay;
    float rho;
    float musX;
    float musY;
    uint valid;
};

float neuralSigmoid(float x)
{
    return 1.f / (1.f + exp(-x));
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
    p.wd = 0.5f;
    p.ws = 0.5f;
    p.ax = 0.5f;
    p.ay = 0.5f;
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

    float m = max(raw[0], raw[3]);
    float e0 = exp(raw[0] - m);
    float e1 = exp(raw[3] - m);
    float wsum = max(e0 + e1, 1e-8f);
    p.wd = e0 / wsum;
    p.ws = e1 / wsum;

    p.muDx = raw[1];
    p.muDy = raw[2];

    const float alphaMin = 0.01f;
    const float alphaMax = 1.f;
    p.ax = clamp(lerp(alphaMin, alphaMax, neuralSigmoid(raw[4])), alphaMin, alphaMax);
    p.ay = clamp(lerp(alphaMin, alphaMax, neuralSigmoid(raw[5])), alphaMin, alphaMax);
    p.rho = clamp(tanh(raw[6]), -0.999f, 0.999f);
    p.musX = raw[7];
    p.musY = raw[8];
    p.valid = 1;
    return p;
}

float evalNeuralSpecularPdfFromParams(const NeuralSamplerParams p, const float3 wiLocal, const float3 woLocal)
{
    if (wiLocal.z <= 0.f || woLocal.z <= 0.f)
        return 0.f;

    float3 wh = neuralSafeNormalize(wiLocal + woLocal);
    if (wh.z < 0.f)
        wh = -wh;

    float s = sqrt(max(1.f - p.rho * p.rho, 1e-8f));
    float ax = max(p.ax, 1e-6f);
    float ay = max(p.ay, 1e-6f);

    float vx = (wh.x + p.musX * wh.z) / ax;
    float vy = (wh.y + p.musY * wh.z - ay * p.rho * vx) / max(ay * s, 1e-8f);
    float vz = wh.z;

    float3 v = float3(vx, vy, vz);
    float vLen = max(length(v), 1e-8f);
    float3 vHat = v / vLen;

    float dStd = max(vHat.z, 0.f) * kInvPi;
    float detInv = 1.f / max(ax * ay * s, 1e-8f);
    float pWh = dStd * detInv / max(vLen * vLen * vLen, 1e-8f);

    float jac = max(4.f * abs(dot(woLocal, wh)), 1e-8f);
    float ps = pWh / jac;

    bool valid = dot(wiLocal, wh) > 0.f && dot(woLocal, wh) > 0.f;
    return valid ? max(ps, 0.f) : 0.f;
}

float evalNeuralPdfFromParams(const NeuralMaterialData md, const NeuralSamplerParams p, const float3 wiLocal, const float3 woLocal)
{
    if (woLocal.z <= 0.f)
        return 0.f;
    if (p.valid == 0)
        return evalCosinePdf(woLocal);

    float3 nd = neuralSafeNormalize(float3(-p.muDx, -p.muDy, 1.f));
    float pd = max(dot(woLocal, nd), 0.f) * kInvPi;
    float ps = evalNeuralSpecularPdfFromParams(p, wiLocal, woLocal);

    float pdf = p.wd * pd + p.ws * ps;
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

    bool chooseSpec = u.x < p.ws;
    float selectDenom = chooseSpec ? max(p.ws, 1e-6f) : max(p.wd, 1e-6f);
    float uSelect = chooseSpec ? (u.x / selectDenom) : ((u.x - p.ws) / selectDenom);
    uSelect = clamp(uSelect, 1e-6f, 1.f - 1e-6f);
    float uAux = clamp(frac(u.y * 0.754877666f + 0.569840296f), 1e-6f, 1.f - 1e-6f);

    if (!chooseSpec)
    {
        float3 nd = neuralSafeNormalize(float3(-p.muDx, -p.muDy, 1.f));
        float3 td;
        float3 bd;
        buildBasisFromNormal(nd, td, bd);

        float3 woLocalSample = sampleCosineHemisphere(float2(uSelect, uAux));
        float3 wo = neuralSafeNormalize(td * woLocalSample.x + bd * woLocalSample.y + nd * woLocalSample.z);
        return wo;
    }

    float s = sqrt(max(1.f - p.rho * p.rho, 1e-8f));
    float3 wStd = sampleCosineHemisphere(float2(uSelect, uAux));
    float3 wh = float3(
        p.ax * wStd.x - p.musX * wStd.z,
        p.ay * p.rho * wStd.x + p.ay * s * wStd.y - p.musY * wStd.z,
        wStd.z
    );
    wh = neuralSafeNormalize(wh);
    if (wh.z < 0.f)
        wh = -wh;

    float3 wo = neuralSafeNormalize(2.f * dot(wiLocal, wh) * wh - wiLocal);
    if (wo.z <= 0.f)
        return sampleCosineHemisphere(u);
    return wo;
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
