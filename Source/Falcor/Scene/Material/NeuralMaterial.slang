import Scene.Scene;
import Scene.Material.MaterialSystem;
import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Utils.Math.ShadingFrame;

static const float kPi = 3.14159265358979323846f;
static const float kInvPi = 0.31830988618379067154f;
static const float expOffset = 3.0f;

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

    uint samplerFrameLinearBufferID = 0;
    uint samplerW0BufferID = 0;
    uint samplerB0BufferID = 0;
    uint samplerW1BufferID = 0;
    uint samplerB1BufferID = 0;
    uint samplerW2BufferID = 0;
    uint samplerB2BufferID = 0;

    uint brdfMlpWidth = 32;
    uint brdfMlpDepth = 2;

    //uint samplerMlpWidth = 32;
    //uint samplerMlpDepth = 2;
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

float3 evalMLP16x2(
    ByteAddressBuffer W0,
    ByteAddressBuffer B0,
    ByteAddressBuffer W1,
    ByteAddressBuffer B1,
    ByteAddressBuffer W2,
    ByteAddressBuffer B2,
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
    ByteAddressBuffer W0,
    ByteAddressBuffer B0,
    ByteAddressBuffer W1,
    ByteAddressBuffer B1,
    ByteAddressBuffer W2,
    ByteAddressBuffer B2,
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
    ByteAddressBuffer W0,
    ByteAddressBuffer B0,
    ByteAddressBuffer W1,
    ByteAddressBuffer B1,
    ByteAddressBuffer W2,
    ByteAddressBuffer B2,
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
    ByteAddressBuffer W0,
    ByteAddressBuffer B0,
    ByteAddressBuffer W1,
    ByteAddressBuffer B1,
    ByteAddressBuffer W2,
    ByteAddressBuffer B2,
    ByteAddressBuffer W3,
    ByteAddressBuffer B3,
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
        if (r == 0)
            y.x = s;
        else if (r == 1)
            y.y = s;
        else
            y.z = s;
    }
    return y;
}

void evalSamplerMLP(
    ByteAddressBuffer W0,
    ByteAddressBuffer B0,
    ByteAddressBuffer W1,
    ByteAddressBuffer B1,
    ByteAddressBuffer W2,
    ByteAddressBuffer B2,
    uint width,
    const float x[14],
    out float y[10]
)
{
    [unroll]
    for (uint i = 0; i < 10; ++i)
        y[i] = 0.f;
    if (width == 0 || width > 64)
        return;

    float h0[64];
    [loop]
    for (uint r = 0; r < width; ++r)
    {
        float s = neuralBufferLoad(B0, r);
        [unroll]
        for (uint c = 0; c < 14; ++c)
        {
            s += neuralBufferLoad(W0, r * 14 + c) * x[c];
        }
        h0[r] = max(s, 0.f);
    }

    float h1[64];
    [loop]
    for (uint r = 0; r < width; ++r)
    {
        float s = neuralBufferLoad(B1, r);
        [loop]
        for (uint c = 0; c < width; ++c)
        {
            s += neuralBufferLoad(W1, r * width + c) * h0[c];
        }
        h1[r] = max(s, 0.f);
    }

    [unroll]
    for (uint r = 0; r < 10; ++r)
    {
        float s = neuralBufferLoad(B2, r);
        [loop]
        for (uint c = 0; c < width; ++c)
        {
            s += neuralBufferLoad(W2, r * width + c) * h1[c];
        }
        y[r] = s;
    }
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

    ByteAddressBuffer frameLinear = materials.getBuffer(md.frameLinearBufferID);
    ByteAddressBuffer W0 = materials.getBuffer(md.W0BufferID);
    ByteAddressBuffer B0 = materials.getBuffer(md.B0BufferID);
    ByteAddressBuffer W1 = materials.getBuffer(md.W1BufferID);
    ByteAddressBuffer B1 = materials.getBuffer(md.B1BufferID);
    ByteAddressBuffer W2 = materials.getBuffer(md.W2BufferID);
    ByteAddressBuffer B2 = materials.getBuffer(md.B2BufferID);
    ByteAddressBuffer W3 = materials.getBuffer(md.W3BufferID);
    ByteAddressBuffer B3 = materials.getBuffer(md.B3BufferID);

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
        y = evalMLP16x2(W0, B0, W1, B1, W2, B2, x);
    else if (md.brdfMlpWidth == 32 && md.brdfMlpDepth == 2)
        y = evalMLP32x2(W0, B0, W1, B1, W2, B2, x);
    else if (md.brdfMlpWidth == 64 && md.brdfMlpDepth == 2)
        y = evalMLP64x2(W0, B0, W1, B1, W2, B2, x);
    else if (md.brdfMlpWidth == 64 && md.brdfMlpDepth == 3)
        y = evalMLP64x3(W0, B0, W1, B1, W2, B2, W3, B3, x);
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
    float3 T0;
    float3 B0;
    float3 N0;
    float3 T1;
    float3 B1;
    float3 N1;
    float alphaX0;
    float alphaY0;
    float alphaX1;
    float alphaY1;
    float weight0;
    float weight1;
    uint valid;
};

float neuralSigmoid(float x)
{
    return 1.f / (1.f + exp(-x));
}

float3 toLobeLocal(const float3 v, const float3 T, const float3 B, const float3 N)
{
    return float3(dot(v, T), dot(v, B), dot(v, N));
}

float3 fromLobeLocal(const float3 vLocal, const float3 T, const float3 B, const float3 N)
{
    return vLocal.x * T + vLocal.y * B + vLocal.z * N;
}

float ggxNdfAniso(const float3 hLocal, float alphaX, float alphaY)
{
    float hz = max(hLocal.z, 0.f);
    if (hz <= 0.f)
        return 0.f;

    float ax = max(alphaX, 1e-6f);
    float ay = max(alphaY, 1e-6f);
    float hx = hLocal.x;
    float hy = hLocal.y;
    float t = (hx / ax) * (hx / ax) + (hy / ay) * (hy / ay) + hz * hz;
    float denomSq = t * t;
    return 1.f / max(kPi * ax * ay * denomSq, 1e-10f);
}

float ggxSmithG1Aniso(const float3 vLocal, float alphaX, float alphaY)
{
    float cosTheta = max(vLocal.z, 1e-6f);
    float sinTheta = sqrt(max(1.f - cosTheta * cosTheta, 0.f));
    float invSin = 1.f / max(sinTheta, 1e-6f);
    float cosPhi = clamp(vLocal.x * invSin, -1.f, 1.f);
    float sinPhi = clamp(vLocal.y * invSin, -1.f, 1.f);
    float alphaEff = sqrt(max((cosPhi * alphaX) * (cosPhi * alphaX) + (sinPhi * alphaY) * (sinPhi * alphaY), 1e-12f));
    float tanTheta = sinTheta / max(cosTheta, 1e-6f);
    float denom = 1.f + max(alphaEff * tanTheta, 0.f);
    return 2.f / (1.f + denom);
}

float ggxVndfPdf(const float3 wiLocal, const float3 mLocal, float alphaX, float alphaY)
{
    float wiDotM = max(dot(wiLocal, mLocal), 1e-6f);
    float D = ggxNdfAniso(mLocal, alphaX, alphaY);
    float G1 = ggxSmithG1Aniso(wiLocal, alphaX, alphaY);
    return G1 * D * wiDotM / max(wiLocal.z, 1e-6f);
}

float3 sampleGgxVndfSingleLobe(const float3 wiLocal, float alphaX, float alphaY, float u1, float u2)
{
    float3 wiS = neuralSafeNormalize(float3(wiLocal.x * alphaX, wiLocal.y * alphaY, max(wiLocal.z, 1e-6f)));

    float signZ = wiS.z >= 0.f ? 1.f : -1.f;
    float a = -1.f / max(signZ + wiS.z, 1e-6f);
    float b = wiS.x * wiS.y * a;

    float3 t1 = float3(1.f + signZ * wiS.x * wiS.x * a, signZ * b, -signZ * wiS.x);
    float3 t2 = float3(b, signZ + wiS.y * wiS.y * a, -wiS.y);

    float r = sqrt(clamp(u1, 0.f, 1.f));
    float phi = 2.f * kPi * clamp(u2, 0.f, 1.f);
    float t = r * cos(phi);
    float s = r * sin(phi);
    if (wiS.z < 0.f)
    {
        float absT = abs(t);
        t = t + s * (1.f - absT) * ((1.f - absT) / max(1.f - abs(wiS.z), 1e-6f));
    }

    float3 mh = t * t1 + s * t2 + sqrt(max(1.f - t * t - s * s, 0.f)) * wiS;
    float3 m = neuralSafeNormalize(float3(mh.x * alphaX, mh.y * alphaY, max(mh.z, 0.f)));
    return m;
}

NeuralSamplerParams evalNeuralSamplerParams(const MaterialSystem materials, const NeuralMaterialData md, const float z[8], float3 wiLocal)
{
    NeuralSamplerParams p;
    p.T0 = float3(1.f, 0.f, 0.f);
    p.B0 = float3(0.f, 1.f, 0.f);
    p.N0 = float3(0.f, 0.f, 1.f);
    p.T1 = p.T0;
    p.B1 = p.B0;
    p.N1 = p.N0;
    p.alphaX0 = 0.5f;
    p.alphaY0 = 0.5f;
    p.alphaX1 = 0.5f;
    p.alphaY1 = 0.5f;
    p.weight0 = 0.5f;
    p.weight1 = 0.5f;
    p.valid = 0;

    if (wiLocal.z <= 0.f)
        return p;

    ByteAddressBuffer frameLinear = materials.getBuffer(md.samplerFrameLinearBufferID);
    ByteAddressBuffer W0 = materials.getBuffer(md.samplerW0BufferID);
    ByteAddressBuffer B0 = materials.getBuffer(md.samplerB0BufferID);
    ByteAddressBuffer W1 = materials.getBuffer(md.samplerW1BufferID);
    ByteAddressBuffer B1 = materials.getBuffer(md.samplerB1BufferID);
    ByteAddressBuffer W2 = materials.getBuffer(md.samplerW2BufferID);
    ByteAddressBuffer B2 = materials.getBuffer(md.samplerB2BufferID);

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

    float x[14];
    [unroll]
    for (uint i = 0; i < 8; ++i)
        x[i] = z[i];

    [unroll]
    for (uint f = 0; f < 2; ++f)
    {
        uint sfBase = 6 * f;
        uint xBase = 8 + 3 * f;

        float3 N = neuralSafeNormalize(float3(sf[sfBase + 0], sf[sfBase + 1], sf[sfBase + 2]));
        float3 Traw = float3(sf[sfBase + 3], sf[sfBase + 4], sf[sfBase + 5]);
        float3 T = neuralSafeNormalize(Traw - dot(Traw, N) * N);
        float3 B = cross(N, T);

        float3 wiPred = float3(dot(wiLocal, T), dot(wiLocal, B), dot(wiLocal, N));
        x[xBase + 0] = wiPred.x;
        x[xBase + 1] = wiPred.y;
        x[xBase + 2] = wiPred.z;
    }

    float raw[10];
    evalSamplerMLP(W0, B0, W1, B1, W2, B2, md.brdfMlpWidth, x, raw);

    float3 N0 = neuralSafeNormalize(float3(sf[0], sf[1], sf[2]));
    float3 T0raw = float3(sf[3], sf[4], sf[5]);
    float3 T0 = neuralSafeNormalize(T0raw - dot(T0raw, N0) * N0);
    float3 B0v = cross(N0, T0);

    float3 N1 = neuralSafeNormalize(float3(sf[6], sf[7], sf[8]));
    float3 T1raw = float3(sf[9], sf[10], sf[11]);
    float3 T1 = neuralSafeNormalize(T1raw - dot(T1raw, N1) * N1);
    float3 B1v = cross(N1, T1);

    const float alphaMin = 0.01f;
    const float alphaMax = 1.f;
    p.alphaX0 = clamp(lerp(alphaMin, alphaMax, neuralSigmoid(raw[0])), alphaMin, alphaMax);
    p.alphaY0 = clamp(lerp(alphaMin, alphaMax, neuralSigmoid(raw[1])), alphaMin, alphaMax);
    p.alphaX1 = clamp(lerp(alphaMin, alphaMax, neuralSigmoid(raw[5])), alphaMin, alphaMax);
    p.alphaY1 = clamp(lerp(alphaMin, alphaMax, neuralSigmoid(raw[6])), alphaMin, alphaMax);

    float m = max(raw[2], raw[7]);
    float e0 = exp(raw[2] - m);
    float e1 = exp(raw[7] - m);
    float wsum = max(e0 + e1, 1e-8f);
    p.weight0 = e0 / wsum;
    p.weight1 = e1 / wsum;

    p.T0 = T0;
    p.B0 = B0v;
    p.N0 = N0;
    p.T1 = T1;
    p.B1 = B1v;
    p.N1 = N1;
    p.valid = 1;
    return p;
}

float evalNeuralPdf(
    const MaterialSystem materials,
    const NeuralMaterialData md,
    const float z[8],
    const float3 wiLocal,
    const float3 woLocal
)
{
    if (woLocal.z <= 0.f)
        return 0.f;

    NeuralSamplerParams p = evalNeuralSamplerParams(materials, md, z, wiLocal);
    if (p.valid == 0)
        return evalCosinePdf(woLocal);

    float3 wi0 = toLobeLocal(wiLocal, p.T0, p.B0, p.N0);
    float3 wo0 = toLobeLocal(woLocal, p.T0, p.B0, p.N0);
    if (wi0.z < 0.f)
        wi0 = float3(wi0.x, wi0.y, -wi0.z);
    float3 m0 = neuralSafeNormalize(wi0 + wo0);
    if (m0.z < 0.f)
        m0 = -m0;
    float woDotM0 = max(abs(dot(wo0, m0)), 1e-6f);
    float pM0 = ggxVndfPdf(wi0, m0, p.alphaX0, p.alphaY0);
    float pWo0 = pM0 / max(4.f * woDotM0, 1e-6f);

    float3 wi1 = toLobeLocal(wiLocal, p.T1, p.B1, p.N1);
    float3 wo1 = toLobeLocal(woLocal, p.T1, p.B1, p.N1);
    if (wi1.z < 0.f)
        wi1 = float3(wi1.x, wi1.y, -wi1.z);
    float3 m1 = neuralSafeNormalize(wi1 + wo1);
    if (m1.z < 0.f)
        m1 = -m1;
    float woDotM1 = max(abs(dot(wo1, m1)), 1e-6f);
    float pM1 = ggxVndfPdf(wi1, m1, p.alphaX1, p.alphaY1);
    float pWo1 = pM1 / max(4.f * woDotM1, 1e-6f);

    float pdf = p.weight0 * max(pWo0, 0.f) + p.weight1 * max(pWo1, 0.f);
    return max(pdf, 1e-10f);
}

float evalNeuralPdfFromParams(const NeuralMaterialData md, const NeuralSamplerParams p, const float3 wiLocal, const float3 woLocal)
{
    if (woLocal.z <= 0.f)
        return 0.f;
    if (p.valid == 0)
        return evalCosinePdf(woLocal);

    float3 wi0 = toLobeLocal(wiLocal, p.T0, p.B0, p.N0);
    float3 wo0 = toLobeLocal(woLocal, p.T0, p.B0, p.N0);
    if (wi0.z < 0.f)
        wi0 = float3(wi0.x, wi0.y, -wi0.z);
    float3 m0 = neuralSafeNormalize(wi0 + wo0);
    if (m0.z < 0.f)
        m0 = -m0;
    float woDotM0 = max(abs(dot(wo0, m0)), 1e-6f);
    float pM0 = ggxVndfPdf(wi0, m0, p.alphaX0, p.alphaY0);
    float pWo0 = pM0 / max(4.f * woDotM0, 1e-6f);

    float3 wi1 = toLobeLocal(wiLocal, p.T1, p.B1, p.N1);
    float3 wo1 = toLobeLocal(woLocal, p.T1, p.B1, p.N1);
    if (wi1.z < 0.f)
        wi1 = float3(wi1.x, wi1.y, -wi1.z);
    float3 m1 = neuralSafeNormalize(wi1 + wo1);
    if (m1.z < 0.f)
        m1 = -m1;
    float woDotM1 = max(abs(dot(wo1, m1)), 1e-6f);
    float pM1 = ggxVndfPdf(wi1, m1, p.alphaX1, p.alphaY1);
    float pWo1 = pM1 / max(4.f * woDotM1, 1e-6f);

    float pdf = p.weight0 * max(pWo0, 0.f) + p.weight1 * max(pWo1, 0.f);
    return max(pdf, 1e-10f);
}

float3 sampleNeuralDirection(const MaterialSystem materials, const NeuralMaterialData md, const float z[8], const float3 wiLocal, float2 u)
{
    NeuralSamplerParams p = evalNeuralSamplerParams(materials, md, z, wiLocal);
    if (p.valid == 0)
        return sampleCosineHemisphere(u);

    bool choose0 = u.x < p.weight0;
    float selectDenom = choose0 ? max(p.weight0, 1e-6f) : max(p.weight1, 1e-6f);
    float uSelect = choose0 ? (u.x / selectDenom) : ((u.x - p.weight0) / selectDenom);
    uSelect = clamp(uSelect, 1e-6f, 1.f - 1e-6f);
    float uAux = clamp(frac(u.y * 0.754877666f + 0.569840296f), 1e-6f, 1.f - 1e-6f);

    float3 T = choose0 ? p.T0 : p.T1;
    float3 B = choose0 ? p.B0 : p.B1;
    float3 N = choose0 ? p.N0 : p.N1;
    float alphaX = choose0 ? p.alphaX0 : p.alphaX1;
    float alphaY = choose0 ? p.alphaY0 : p.alphaY1;

    float3 wiLobe = toLobeLocal(wiLocal, T, B, N);
    if (wiLobe.z < 0.f)
        wiLobe = float3(wiLobe.x, wiLobe.y, -wiLobe.z);

    float3 mLobe = sampleGgxVndfSingleLobe(wiLobe, alphaX, alphaY, uSelect, uAux);
    float3 woLobe = 2.f * dot(wiLobe, mLobe) * mLobe - wiLobe;

    float3 wo = neuralSafeNormalize(fromLobeLocal(woLobe, T, B, N));
    if (wo.z <= 0.f)
    {
        return sampleCosineHemisphere(u);
    }
    return wo;
}

float3 sampleNeuralDirectionFromParams(const NeuralMaterialData md, const NeuralSamplerParams p, const float3 wiLocal, float2 u)
{
    if (p.valid == 0)
        return sampleCosineHemisphere(u);

    bool choose0 = u.x < p.weight0;
    float selectDenom = choose0 ? max(p.weight0, 1e-6f) : max(p.weight1, 1e-6f);
    float uSelect = choose0 ? (u.x / selectDenom) : ((u.x - p.weight0) / selectDenom);
    uSelect = clamp(uSelect, 1e-6f, 1.f - 1e-6f);
    float uAux = clamp(frac(u.y * 0.754877666f + 0.569840296f), 1e-6f, 1.f - 1e-6f);

    float3 T = choose0 ? p.T0 : p.T1;
    float3 B = choose0 ? p.B0 : p.B1;
    float3 N = choose0 ? p.N0 : p.N1;
    float alphaX = choose0 ? p.alphaX0 : p.alphaX1;
    float alphaY = choose0 ? p.alphaY0 : p.alphaY1;

    float3 wiLobe = toLobeLocal(wiLocal, T, B, N);
    if (wiLobe.z < 0.f)
        wiLobe = float3(wiLobe.x, wiLobe.y, -wiLobe.z);

    float3 mLobe = sampleGgxVndfSingleLobe(wiLobe, alphaX, alphaY, uSelect, uAux);
    float3 woLobe = 2.f * dot(wiLobe, mLobe) * mLobe - wiLobe;
    float3 wo = neuralSafeNormalize(fromLobeLocal(woLobe, T, B, N));

    if (wo.z <= 0.f)
    {
        return sampleCosineHemisphere(u);
    }
    return wo;
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
