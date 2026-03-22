#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Scene.Material.ShadingUtils;
import Rendering.Materials.RGLCommon;
import Utils.Sampling.UniformSampleGenerator;
import Rendering.Materials.StandardMaterial;
import BSDFOptimizerParams;
import BSDFOptimizerHelpers;

struct BsdfSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
    float3 specular;
    float3 albedo;
    float3 roughness;
    float3 normal;
};

struct BsdfTestSampleData
{
    float2 uv;
    float3 wo;
    float3 wi;
    float3 f;
};

RWStructuredBuffer<BsdfTestSampleData> gSampleOutputBuffer;
uint gSampleCount;

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

float3 uniformSampleSphere(float2 u)
{
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z*z));
    float phi = 2.0 * M_PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gSampleCount) return;

    //Sample UV and view direction uniformly
    uint seed = idx * 9781 + 6271;

    float2 uv = float2(rand(seed), rand(seed));

    float3 wo = sample_hemisphere(float2(rand(seed), rand(seed)));
    float3 wi = sample_hemisphere(float2(rand(seed), rand(seed)));

    //Construct Shadign data and samplers
    UniformSampleGenerator sg = UniformSampleGenerator(1, 0);
    let lod = ExplicitLodTextureSampler(0.f);

    //Shading data should be prepared based on uniformly sampled UV value, used to reconstruct w_o
    ShadingData sd = {};
    sd.posW = float3(uv, 0.0);
    sd.faceN = float3(0, 0, 1);
    sd.V = wi;
    sd.frame = ShadingFrame::createIdentity();

    //Get material Instance
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    BSDFProperties props = mi.getProperties(sd);

    // For some fucking reason the ray towards the surface (w_i) is called wo, V is the view Vector
    float3 f = mi.eval(sd, wo, sg);

    //Save results
    BsdfTestSampleData data;
    data.uv = uv;
    data.wi = sd.V;
    data.wo = wo;
    data.f = f;
    //data.albedo = props.diffuseReflectionAlbedo;
    //data.specular = props.specularReflectionAlbedo;
    //data.roughness = props.roughness;
    //data.normal = sd.faceN;

    gSampleOutputBuffer[idx] = data;
}
