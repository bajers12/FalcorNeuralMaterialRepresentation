#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.Shading;
import Scene.Material.MaterialFactory;
import Utils.Math.MathHelpers;
import Utils.Sampling.UniformSampleGenerator;

struct BsdfSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
    float3 specular;
    float3 albedo;
    float3 normal;
    float roughness;
    float pdf;
};

RWStructuredBuffer<BsdfSampleData> gSampleOutputBuffer;
uniform uint gSampleCount;
uniform uint gMaterialId;
uniform uint gRunSeed;
uniform uint gSeedDomain;
uniform uint gGenerationIndex;
uniform bool gUseUvGrid;
uniform uint gUvGridFullWidth;
uniform uint gUvGridFullHeight;

uint hash(uint x)
{
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

float rand(inout uint seed)
{
    seed = hash(seed);
    return float(seed) / 4294967296.0;
}

uint mixSeed(uint a, uint b)
{
    return hash(a ^ hash(b + 0x9e3779b9u));
}

uint makeSampleSeed(uint sampleIdx, uint streamId)
{
    uint seed = mixSeed(gRunSeed, gSeedDomain);
    seed = mixSeed(seed, gGenerationIndex);
    seed = mixSeed(seed, sampleIdx);
    seed = mixSeed(seed, streamId);
    return seed;
}

[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint idx = threadId.x;

    uint uvSeed = makeSampleSeed(idx, 0u);
    uint woSeed = makeSampleSeed(idx, 1u);
    uint wiSeed = makeSampleSeed(idx, 2u);
    uint sgSeed = makeSampleSeed(idx, 3u);

    float2 uv = float2(rand(uvSeed), rand(uvSeed));
    if (gUseUvGrid)
    {
        uint2 texel = uint2(idx % gUvGridFullWidth, idx / gUvGridFullWidth);
        uv = (float2(texel) + 0.5f) / float2(gUvGridFullWidth, gUvGridFullHeight);
    }

    float3 wo = sample_hemisphere(float2(rand(woSeed), rand(woSeed)));
    float3 wi = sample_hemisphere(float2(rand(wiSeed), rand(wiSeed)));

    // Keep independent sample streams for UVs, BSDF directions, and any
    // additional material sampling performed through the Falcor interfaces.
    UniformSampleGenerator sg = UniformSampleGenerator(sgSeed, 0);
    let lod = ExplicitLodTextureSampler(0.f);

    VertexData v = {};
    v.posW        = float3((uv[0]-0.5f)*2, (uv[1]-0.5f)*2, 0.f);
    v.normalW     = float3(0.f, 0.f, 1.f);         // flat geometric normal
    v.tangentW    = float4(1.f, 0.f, 0.f, 1.f);
    v.faceNormalW = v.normalW;
    v.texC        = uv;

    ShadingData sd = gScene.materials.prepareShadingData(v, gMaterialId, wi);

    const uint materialHints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, materialHints);
    BSDFProperties props = mi.getProperties(sd);

    // For some fucking reason the ray towards the surface (w_i) is called wo, V is the view Vector
    float3 f = mi.eval(sd, wo, sg);
    float pdf = 0.0f;

    // Save results
    BsdfSampleData data;
    data.uv       = uv;
    data.wi       = wi;   // incident/view direction
    data.wo       = wo;   // sampled light direction
    data.f        = f;
    data.albedo   = props.diffuseReflectionAlbedo;
    data.specular = props.specularReflectionAlbedo;
    data.normal = props.guideNormal;
    data.roughness = props.roughness;
    data.pdf = pdf;

    gSampleOutputBuffer[idx] = data;
}
