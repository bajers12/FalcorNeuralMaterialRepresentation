cbuffer Params
{
    uint2 gOutputDim;
    float gExpOffset;
    uint  gApplyExp;
    uint  gFlipY;
    uint  gDisplayMode;      // 0=raw,1=clamp01,2=log1p(max(x,0)),3=abs
    uint  gDebugTensor;      // see enum below
    uint  gDebugBaseChannel; // first displayed channel

    uint gDumpSamplePixels; // 0 or 1
    uint2 gDebugCoords[kDebugSampleCount];
};

// Input latent textures
Texture2D<float4> gLatent0;
Texture2D<float4> gLatent1;

// Output
RWTexture2D<float4> gOutput;

// Decoder weights
StructuredBuffer<float> gFrameLinear; // 12*8
StructuredBuffer<float> gW0;          // 32*20
StructuredBuffer<float> gB0;          // 32
StructuredBuffer<float> gW1;          // 32*32
StructuredBuffer<float> gB1;          // 32
StructuredBuffer<float> gW2;          // 3*32
StructuredBuffer<float> gB2;          // 3

RWStructuredBuffer<float4> gDebugSamples;

float relu(float x)
{
    return max(x, 0.0);
}

float linearRead(StructuredBuffer<float> buf, uint row, uint col, uint cols)
{
    return buf[row * cols + col];
}

float3 applyDisplayMode(float3 v, uint mode)
{
    switch (mode)
    {
    default:
    case 0: // raw
        return v;

    case 1: // clamp01
        return clamp(v, 0.0, 1.0);

    case 2: // log1p(max(x,0))
        return log(1.0 + max(v, 0.0));

    case 3: // abs
        return abs(v);
    }
}

float3 load3(const float arr[], uint size, uint base)
{
    float3 v = float3(0.0, 0.0, 0.0);
    if (base + 0 < size) v.x = arr[base + 0];
    if (base + 1 < size) v.y = arr[base + 1];
    if (base + 2 < size) v.z = arr[base + 2];
    return v;
}

static const uint kDebugSampleCount = 5;

void maybeWriteDebugSample(uint2 tid, float3 value)
{
    if (gDumpSamplePixels == 0) return;

    [unroll]
    for (uint i = 0; i < kDebugSampleCount; ++i)
    {
        if (all(tid == gDebugCoords[i]))
        {
            gDebugSamples[i] = float4(value, 1.0);
        }
    }
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gOutputDim.x || tid.y >= gOutputDim.y) return;

    uint2 samplePos = tid.xy;
    if (gFlipY != 0)
    {
        samplePos.y = gOutputDim.y - 1 - samplePos.y;
    }

    float4 z0 = gLatent0[samplePos];
    float4 z1 = gLatent1[samplePos];

    float z[8];
    z[0] = z0.x; z[1] = z0.y; z[2] = z0.z; z[3] = z0.w;
    z[4] = z1.x; z[5] = z1.y; z[6] = z1.z; z[7] = z1.w;

    float frame[12];
    [unroll]
    for (uint r = 0; r < 12; ++r)
    {
        float s = 0.0;
        [unroll]
        for (uint c = 0; c < 8; ++c)
        {
            s += linearRead(gFrameLinear, r, c, 8) * z[c];
        }
        frame[r] = s;
    }

    float x[20];
    [unroll]
    for (uint i = 0; i < 8; ++i) x[i] = z[i];
    [unroll]
    for (uint i = 0; i < 12; ++i) x[8 + i] = frame[i];

    float h0pre[32];
    float h0[32];
    [unroll]
    for (uint r = 0; r < 32; ++r)
    {
        float s = gB0[r];
        [unroll]
        for (uint c = 0; c < 20; ++c)
        {
            s += linearRead(gW0, r, c, 20) * x[c];
        }
        h0pre[r] = s;
        h0[r] = relu(s);
    }

    float h1pre[32];
    float h1[32];
    [unroll]
    for (uint r = 0; r < 32; ++r)
    {
        float s = gB1[r];
        [unroll]
        for (uint c = 0; c < 32; ++c)
        {
            s += linearRead(gW1, r, c, 32) * h0[c];
        }
        h1pre[r] = s;
        h1[r] = relu(s);
    }

    float yArr[3];
    [unroll]
    for (uint r = 0; r < 3; ++r)
    {
        float s = gB2[r];
        [unroll]
        for (uint c = 0; c < 32; ++c)
        {
            s += linearRead(gW2, r, c, 32) * h1[c];
        }
        yArr[r] = s;
    }

    float3 y = float3(yArr[0], yArr[1], yArr[2]);
    float3 decoded = (gApplyExp != 0) ? exp(y - gExpOffset) : y;

    float3 debugValue = float3(0.0);

    switch (gDebugTensor)
    {
    default:
    case 0: // final decoded
        debugValue = decoded;
        break;

    case 1: // raw y
        debugValue = y;
        break;

    case 2: // z
        debugValue = load3(z, 8, gDebugBaseChannel);
        break;

    case 3: // frame
        debugValue = load3(frame, 12, gDebugBaseChannel);
        break;

    case 4: // x
        debugValue = load3(x, 20, gDebugBaseChannel);
        break;

    case 5: // h0 pre-ReLU
        debugValue = load3(h0pre, 32, gDebugBaseChannel);
        break;

    case 6: // h0 post-ReLU
        debugValue = load3(h0, 32, gDebugBaseChannel);
        break;

    case 7: // h1 pre-ReLU
        debugValue = load3(h1pre, 32, gDebugBaseChannel);
        break;

    case 8: // h1 post-ReLU
        debugValue = load3(h1, 32, gDebugBaseChannel);
        break;
    }

    float3 display = applyDisplayMode(debugValue, gDisplayMode);
    gOutput[tid.xy] = float4(display, 1.0);
    maybeWriteDebugSample(tid.xy, debugValue);
}