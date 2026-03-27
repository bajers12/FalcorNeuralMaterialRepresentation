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
};

struct BsdfTestSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
};

RWStructuredBuffer<BsdfSampleData> gSampleOutputBuffer;
uint gSampleCount;
uint gMaterialId = 3;

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

[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint idx = threadId.x;

    //Sample UV and view direction uniformly
    uint seed = idx * 9781 + 6271;

    float2 uv = float2(rand(seed), rand(seed));

    float3 wo = sample_hemisphere(float2(rand(seed), rand(seed)));
    float3 wi = sample_hemisphere(float2(rand(seed), rand(seed)));

    // Construct Shadign data and samplers
    UniformSampleGenerator sg = UniformSampleGenerator(idx, 0);
    let lod = ExplicitLodTextureSampler(0.f);

    //Shading data should be prepared based on uniformly sampled UV value, used to reconstruct w_o
    ShadingData sd = {};
    sd.posW = float3(0.0, 0.0, 0.0);
    sd.V = wi;  // V is the view direction (outgoing)
    sd.uv = uv;
    sd.frame = ShadingFrame::createIdentity();
    sd.faceN = sd.frame.N;
    sd.frontFacing = true;

    sd.mtl = gScene.materials.getMaterialHeader(gMaterialId);
    sd.materialID = gMaterialId;
    sd.IoR = 1.f;

    let mi = gScene.materials.getMaterialInstance(sd, lod);
    BSDFProperties props = mi.getProperties(sd);

    // For some fucking reason the ray towards the surface (w_i) is called wo, V is the view Vector
    float3 f = mi.eval(sd, wo, sg);

    // Save results
    BsdfSampleData data;
    data.uv = uv;
    data.wi = wi;  // incident direction
    data.wo = wo;  // outgoing direction
    data.f = f;
    data.albedo = props.diffuseReflectionAlbedo;
    data.specular = props.specularReflectionAlbedo;
    data.normal = sd.faceN;
    //Can also use normal after applied normal maps.
    //data.normal = props.guideNormal;
    data.roughness = props.roughness;

    gSampleOutputBuffer[idx] = data;
}
