#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.Shading;
import Scene.Material.BootstrapFeatureMaterial;
import Scene.Material.MaterialFactory;
import Utils.Math.MathHelpers;
import Utils.Sampling.UniformSampleGenerator;

#ifndef SAMPLE_FEATURES_ENABLED
#define SAMPLE_FEATURES_ENABLED 0
#endif

#ifndef LEGACY_FEATURES_ENABLED
#define LEGACY_FEATURES_ENABLED 0
#endif

#ifndef MATERIAL_FEATURES_ENABLED
#define MATERIAL_FEATURES_ENABLED 0
#endif

struct BsdfSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
#if SAMPLE_FEATURES_ENABLED
    float4 bootstrapFeature0;
    float4 bootstrapFeature1;
    float4 bootstrapFeature2;
    float4 bootstrapFeature3;
    float4 bootstrapFeature4;
    float4 bootstrapFeature5;
#endif
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
uniform float gMollificationConeAngleRad;
uniform uint gMollificationSampleCount;

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

void makeBasis(float3 n, out float3 tangent, out float3 bitangent)
{
    float3 up = abs(n.z) < 0.999f ? float3(0.f, 0.f, 1.f) : float3(1.f, 0.f, 0.f);
    tangent = normalize(cross(up, n));
    bitangent = cross(n, tangent);
}

float3 sampleConeAround(float3 center, float2 u, float cosTheta)
{
    center = normalize(center);
    float3 coneDir = sample_cone(u, cosTheta);

    float3 tangent;
    float3 bitangent;
    makeBasis(center, tangent, bitangent);

    return coneDir.x * tangent + coneDir.y * bitangent + coneDir.z * center;
}

#if SAMPLE_FEATURES_ENABLED
void clearBootstrapFeatures(inout BsdfSampleData data)
{
    data.bootstrapFeature0 = 0.f.xxxx;
    data.bootstrapFeature1 = 0.f.xxxx;
    data.bootstrapFeature2 = 0.f.xxxx;
    data.bootstrapFeature3 = 0.f.xxxx;
    data.bootstrapFeature4 = 0.f.xxxx;
    data.bootstrapFeature5 = 0.f.xxxx;
}

#if LEGACY_FEATURES_ENABLED
void writeLegacyBootstrapFeatures(const BSDFProperties props, const float pdf, inout BsdfSampleData data)
{
    data.bootstrapFeature0 = float4(props.specularReflectionAlbedo, props.diffuseReflectionAlbedo.x);
    data.bootstrapFeature1 = float4(props.diffuseReflectionAlbedo.yz, props.guideNormal.xy);
    data.bootstrapFeature2 = float4(props.guideNormal.z, props.roughness, pdf, 0.f);
}
#endif

#if MATERIAL_FEATURES_ENABLED
void writeMaterialBootstrapFeatures(const ShadingData sd, inout BsdfSampleData data)
{
    let material = gScene.materials.getBootstrapFeatureMaterial(gMaterialId);
    BootstrapFeatureData features = material.getBootstrapFeatures(gScene.materials, sd);
    data.bootstrapFeature0 = features.feature0;
    data.bootstrapFeature1 = features.feature1;
    data.bootstrapFeature2 = features.feature2;
    data.bootstrapFeature3 = features.feature3;
    data.bootstrapFeature4 = features.feature4;
    data.bootstrapFeature5 = features.feature5;
}
#endif
#endif

[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= gSampleCount) return;

    uint uvSeed = makeSampleSeed(idx, 0u);
    uint woSeed = makeSampleSeed(idx, 1u);
    uint wiSeed = makeSampleSeed(idx, 2u);
    uint sgSeed = makeSampleSeed(idx, 3u);
    uint mollificationSeed = makeSampleSeed(idx, 4u);

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

    // For some fucking reason the ray towards the surface (w_i) is called wo, V is the view Vector
    float3 f = float3(0.f);
    if (gMollificationConeAngleRad > 0.f && gMollificationSampleCount > 1u)
    {
        float cosTheta = cos(min(gMollificationConeAngleRad, M_PI));
        uint validSampleCount = 0u;

        for (uint i = 0u; i < gMollificationSampleCount; ++i)
        {
            float3 mollifiedWo = sampleConeAround(wo, float2(rand(mollificationSeed), rand(mollificationSeed)), cosTheta);
            if (mollifiedWo.z > 0.f)
            {
                f += mi.eval(sd, mollifiedWo, sg);
                validSampleCount++;
            }
        }

        f = validSampleCount > 0u ? f / float(validSampleCount) : mi.eval(sd, wo, sg);
    }
    else
    {
        f = mi.eval(sd, wo, sg);
    }
    float pdf = 0.0f;

    // Save results
    BsdfSampleData data;
    data.uv       = uv;
    data.wi       = wi;   // incident/view direction, towards viewer
    data.wo       = wo;   // sampled light direction, away from surface
    data.f        = f;

#if SAMPLE_FEATURES_ENABLED
    clearBootstrapFeatures(data);

#if LEGACY_FEATURES_ENABLED
    BSDFProperties props = mi.getProperties(sd);
    writeLegacyBootstrapFeatures(props, pdf, data);
#endif

#if MATERIAL_FEATURES_ENABLED
    writeMaterialBootstrapFeatures(sd, data);
#endif
#endif

    gSampleOutputBuffer[idx] = data;
}
