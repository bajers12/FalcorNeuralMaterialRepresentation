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
    float mipLevel;
#if SAMPLE_FEATURES_ENABLED
    float4 bootstrapFeature0;
    float4 bootstrapFeature1;
    float4 bootstrapFeature2;
    float4 bootstrapFeature3;
    float4 bootstrapFeature4;
    float4 bootstrapFeature5;
    float4 bootstrapFeature6;
    float4 bootstrapFeature7;
#endif
};

RWStructuredBuffer<BsdfSampleData> gSampleOutputBuffer;
StructuredBuffer<float2> gUvSampleBuffer;
uniform uint gSampleCount;
uniform uint gMaterialId;
uniform uint gRunSeed;
uniform uint gSeedDomain;
uniform uint gGenerationIndex;
uniform bool gUseUvGrid;
uniform bool gUseUvSamples;
uniform uint gUvGridFullWidth;
uniform uint gUvGridFullHeight;
uniform float gMollificationConeAngleRad;
uniform uint gMollificationSampleCount;
uniform bool gHierarchicalFilteringEnabled;
uniform uint gHierarchicalMipCount;
uniform uint2 gFinestTextureSize;
uniform float gMipExponentialRate;
uniform uint gMinFilterSampleCount;
uniform uint gMaxFilterSampleCount;
uniform float gGaussianFilterStdScale;

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

uint sampleTrainingMip(inout uint seed)
{
    uint mipCount = max(1u, gHierarchicalMipCount);
    if (!gHierarchicalFilteringEnabled || mipCount <= 1u)
        return 0u;

    float total = 0.f;
    [loop]
    for (uint mip = 0u; mip < mipCount; ++mip)
        total += exp(-gMipExponentialRate * float(mip));

    float target = rand(seed) * total;
    float accum = 0.f;
    [loop]
    for (uint mip = 0u; mip < mipCount; ++mip)
    {
        accum += exp(-gMipExponentialRate * float(mip));
        if (target <= accum)
            return mip;
    }
    return mipCount - 1u;
}

uint spatialFilterSampleCount(uint mipLevel)
{
    if (!gHierarchicalFilteringEnabled || mipLevel == 0u)
        return 1u;

    uint count = 1u << min(30u, 2u * mipLevel);
    return clamp(count, max(1u, gMinFilterSampleCount), max(gMinFilterSampleCount, gMaxFilterSampleCount));
}

float gaussian01(inout uint seed)
{
    float u1 = max(rand(seed), 1e-6f);
    float u2 = rand(seed);
    return sqrt(-2.f * log(u1)) * cos(M_2PI * u2);
}

float2 sampleFilteredUv(float2 centerUv, uint mipLevel, inout uint seed)
{
    if (!gHierarchicalFilteringEnabled || mipLevel == 0u || gGaussianFilterStdScale <= 0.f)
        return centerUv;

    float footprint = exp2(float(mipLevel));
    float2 sigma = gGaussianFilterStdScale * footprint / max(float2(gFinestTextureSize), 1.f.xx);
    float2 offset = float2(gaussian01(seed), gaussian01(seed)) * sigma;
    return frac(centerUv + offset);
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
    data.bootstrapFeature6 = 0.f.xxxx;
    data.bootstrapFeature7 = 0.f.xxxx;
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
    data.bootstrapFeature6 = features.feature6;
    data.bootstrapFeature7 = features.feature7;
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
    uint mipSeed = makeSampleSeed(idx, 5u);
    uint filterSeed = makeSampleSeed(idx, 6u);

    float2 uv = float2(rand(uvSeed), rand(uvSeed));
    if (gUseUvSamples)
    {
        uv = gUvSampleBuffer[idx];
    }
    else if (gUseUvGrid)
    {
        uint2 texel = uint2(idx % gUvGridFullWidth, idx / gUvGridFullWidth);
        uv = (float2(texel) + 0.5f) / float2(gUvGridFullWidth, gUvGridFullHeight);
    }

    // Sample wi and wo using the half/difference vector parametrization (Rusinkiewicz 1998).
    //
    // We sample the half vector h and difference direction freely, then reject
    // only candidates where either resulting direction leaves the upper
    // hemisphere. This covers the full valid half/difference domain instead of
    // using the conservative theta_h + theta_d <= pi/2 shortcut.
    //
    // Seed stream 1 (woSeed) drives (phi_h, phi_d), the azimuthal angles.
    // Seed stream 2 (wiSeed) drives theta_d and theta_h, both sampled uniformly
    // over their angular domains [0, pi/2).
    //
    // Note: in this file wi = view/incident direction (toward viewer),
    //       wo = light direction (away from surface). Both must have z > 0.
    const float kHorizonEps = 1e-4f;
    const uint kMaxDirectionAttempts = 64u;

    float3 wi = float3(0.f, 0.f, 1.f);
    float3 wo = float3(0.f, 0.f, 1.f);

    for (uint attempt = 0u; attempt < kMaxDirectionAttempts; ++attempt)
    {
        // --- Half vector h in tangent space (normal = (0,0,1)) ---
        float theta_h = acos(kHorizonEps) * rand(wiSeed);
        float cosTheta_h = cos(theta_h);
        float sinTheta_h = sqrt(max(0.f, 1.f - cosTheta_h * cosTheta_h));
        float phi_h = M_2PI * rand(woSeed);

        float3 h = float3(sinTheta_h * cos(phi_h),
                          sinTheta_h * sin(phi_h),
                          cosTheta_h);

        // --- Build an orthonormal frame with h as the north pole ---
        // Used to express wi (= omega_i in difference-vector frame) relative to h.
        float3 h_tangent, h_bitangent;
        makeBasis(h, h_tangent, h_bitangent);

        // --- Difference direction around h ---
        float theta_d = acos(kHorizonEps) * rand(wiSeed);
        float cosTheta_d = cos(theta_d);
        float sinTheta_d = sqrt(max(0.f, 1.f - cosTheta_d * cosTheta_d));
        // Reciprocity makes phi_d and phi_d + pi equivalent, so sample the
        // canonical Rusinkiewicz difference-azimuth domain [0, pi).
        float phi_d = M_PI * rand(woSeed);

        float3 candidateWi = sinTheta_d * cos(phi_d) * h_tangent
                           + sinTheta_d * sin(phi_d) * h_bitangent
                           + cosTheta_d              * h;

        float3 candidateWo = 2.f * dot(candidateWi, h) * h - candidateWi;

        if (candidateWi.z > kHorizonEps && candidateWo.z > kHorizonEps)
        {
            wi = normalize(candidateWi);
            wo = normalize(candidateWo);
            break;
        }
    }

    // Keep independent sample streams for UVs, BSDF directions, and any
    // additional material sampling performed through the Falcor interfaces.
    uint trainingMip = sampleTrainingMip(mipSeed);
    uint spatialSampleCount = spatialFilterSampleCount(trainingMip);
    // Filtering is owned by this pass through spatial footprint samples. Keep
    // the material's own texture sampling at its existing default LOD behavior.
    let lod = ExplicitLodTextureSampler(0.f);
    const uint materialHints = (uint)MaterialInstanceHints::AdjustShadingNormal;

    // For some fucking reason the ray towards the surface (w_i) is called wo, V is the view Vector
    float3 f = float3(0.f);
    uint validSpatialSampleCount = 0u;
    [loop]
    for (uint spatialSample = 0u; spatialSample < spatialSampleCount; ++spatialSample)
    {
        float2 filteredUv = sampleFilteredUv(uv, trainingMip, filterSeed);
        VertexData v = {};
        v.posW        = float3((filteredUv[0]-0.5f)*2, (filteredUv[1]-0.5f)*2, 0.f);
        v.normalW     = float3(0.f, 0.f, 1.f);         // flat geometric normal
        v.tangentW    = float4(1.f, 0.f, 0.f, 1.f);
        v.faceNormalW = v.normalW;
        v.texC        = filteredUv;

        ShadingData sd = gScene.materials.prepareShadingData(v, gMaterialId, wi);
        let mi = gScene.materials.getMaterialInstance(sd, lod, materialHints);
        UniformSampleGenerator sg = UniformSampleGenerator(sgSeed, spatialSample);

        if (gMollificationConeAngleRad > 0.f && gMollificationSampleCount > 1u)
        {
            float cosTheta = cos(min(gMollificationConeAngleRad, M_PI));
            uint validSampleCount = 0u;
            float3 mollifiedF = 0.f.xxx;

            for (uint i = 0u; i < gMollificationSampleCount; ++i)
            {
                float3 mollifiedWo = sampleConeAround(wo, float2(rand(mollificationSeed), rand(mollificationSeed)), cosTheta);
                if (mollifiedWo.z > 0.f)
                {
                    mollifiedF += mi.eval(sd, mollifiedWo, sg);
                    validSampleCount++;
                }
            }

            f += (validSampleCount >= 3) ? mollifiedF / float(validSampleCount) : mi.eval(sd, wo, sg);
        }
        else
        {
            f += mi.eval(sd, wo, sg);
        }
        validSpatialSampleCount++;
    }
    f /= float(max(1u, validSpatialSampleCount));
    float pdf = 0.0f;

    // Save results
    BsdfSampleData data;
    data.uv       = uv;
    data.wi       = wi;   // incident/view direction, towards viewer
    data.wo       = wo;   // sampled light direction, away from surface
    data.f        = f;
    data.mipLevel = float(trainingMip);

    #if SAMPLE_FEATURES_ENABLED
        clearBootstrapFeatures(data);

        VertexData featureV = {};
        featureV.posW        = float3((uv[0]-0.5f)*2, (uv[1]-0.5f)*2, 0.f);
        featureV.normalW     = float3(0.f, 0.f, 1.f);
        featureV.tangentW    = float4(1.f, 0.f, 0.f, 1.f);
        featureV.faceNormalW = featureV.normalW;
        featureV.texC        = uv;
        ShadingData featureSd = gScene.materials.prepareShadingData(featureV, gMaterialId, wi);
        let featureMi = gScene.materials.getMaterialInstance(featureSd, lod, materialHints);

        #if LEGACY_FEATURES_ENABLED
            BSDFProperties props = featureMi.getProperties(featureSd);
            writeLegacyBootstrapFeatures(props, pdf, data);
        #endif

        #if MATERIAL_FEATURES_ENABLED
            writeMaterialBootstrapFeatures(featureSd, data);
        #endif
    #endif

    gSampleOutputBuffer[idx] = data;
}
