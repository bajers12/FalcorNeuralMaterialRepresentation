import Scene.Scene;
import Scene.Material.BootstrapFeatureMaterial;
import Scene.Material.MaterialSystem;
import Scene.Material.TrainingDirectionGuardMaterial;
import Scene.Material.ShadingUtils;
import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Utils.Math.MathHelpers;
import Utils.Math.ShadingFrame;

static const float kDustOnlyInvPi = 0.31830988618379067154f;
static const float3 kDustOnlyColor = float3(0.55f, 0.52f, 0.46f);
static const float kDustOnlyRoughness = 0.8f;

struct ThreeLayeredGGXDustOnlyMaterialData
{
    TextureHandle texBaseColor = {};
    TextureHandle texNormal = {};
    TextureHandle texLayerRoughness = {};
    TextureHandle texDustCoverage = {};

    float f0 = 0.f;
    float roughnessScale = 1.f;
    float roughnessBias = 0.02f;
    float normalFlatten = 0.8f;

    uint flipNormalY = 1;
};

struct DustOnlyInputs
{
    float coverage;
    float3 normalTS;
};

DustOnlyInputs sampleDustOnlyInputsAtUV(const ThreeLayeredGGXDustOnlyMaterialData data, const uint samplerID, const float2 uv)
{
    SamplerState s = gScene.materials.getTextureSampler(samplerID);
    ExplicitLodTextureSampler lod = { 0.f };

    DustOnlyInputs inputs;
    inputs.coverage = saturate(gScene.materials.sampleTexture(data.texDustCoverage, s, uv, float4(0.f), lod).r);
    inputs.normalTS = gScene.materials.sampleTexture(data.texNormal, s, uv, float4(0.5f, 0.5f, 1.f, 1.f), lod).xyz * 2.f - 1.f;
    if (data.flipNormalY != 0) inputs.normalTS.y = -inputs.normalTS.y;
    return inputs;
}

DustOnlyInputs sampleDustOnlyInputs(const ThreeLayeredGGXDustOnlyMaterialData data, const uint samplerID, const ShadingData sd)
{
    return sampleDustOnlyInputsAtUV(data, samplerID, sd.uv);
}

float3 dustOnlyFlattenNormal(float3 n, float flatten)
{
    n.xy *= 1.f - flatten;
    n.z = max(n.z, 1e-4f);
    return normalize(n);
}

ShadingFrame dustOnlyShadingFrame(const ThreeLayeredGGXDustOnlyMaterialData data, const ShadingData sd, const DustOnlyInputs inputs)
{
    float3 mapN = dustOnlyFlattenNormal(inputs.normalTS, saturate(data.normalFlatten));
    ShadingFrame sf = ShadingFrame(sd.frame.fromLocal(normalize(mapN)), sd.tangentW);
    flipShadingNormal(sd, sf);
    adjustShadingNormal(sd, sf);
    return sf;
}

float3 dustOnlyEval(const DustOnlyInputs inputs, const float3 wiLocal, const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f.xxx;

    float sinThetaI = sqrt(max(0.f, 1.f - wiLocal.z * wiLocal.z));
    float sinThetaO = sqrt(max(0.f, 1.f - woLocal.z * woLocal.z));
    float sinPhiI = sinThetaI == 0.f ? 0.f : clamp(wiLocal.y / sinThetaI, -1.f, 1.f);
    float cosPhiI = sinThetaI == 0.f ? 0.f : clamp(wiLocal.x / sinThetaI, -1.f, 1.f);
    float sinPhiO = sinThetaO == 0.f ? 0.f : clamp(woLocal.y / sinThetaO, -1.f, 1.f);
    float cosPhiO = sinThetaO == 0.f ? 0.f : clamp(woLocal.x / sinThetaO, -1.f, 1.f);
    float cosDiff = max(cosPhiI * cosPhiO + sinPhiI * sinPhiO, 0.f);

    float sigma2 = kDustOnlyRoughness * kDustOnlyRoughness;
    float A = 1.f - 0.5f * sigma2 / (sigma2 + 0.33f);
    float B = 0.45f * sigma2 / (sigma2 + 0.09f);
    float sinAlpha;
    float tanBeta;
    if (abs(wiLocal.z) > abs(woLocal.z))
    {
        sinAlpha = sinThetaO;
        tanBeta = sinThetaI / abs(wiLocal.z);
    }
    else
    {
        sinAlpha = sinThetaI;
        tanBeta = sinThetaO / abs(woLocal.z);
    }

    return inputs.coverage * kDustOnlyColor * kDustOnlyInvPi * (A + B * cosDiff * sinAlpha * tanBeta) * woLocal.z;
}

float dustOnlyPdf(const float3 woLocal)
{
    return woLocal.z > kMinCosTheta ? woLocal.z * kDustOnlyInvPi : 0.f;
}

struct ThreeLayeredGGXDustOnlyMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ThreeLayeredGGXDustOnlyMaterialData data = {};
    uint defaultSamplerID = 0;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        DustOnlyInputs inputs = sampleDustOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = dustOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);
        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f.xxx;
        return dustOnlyEval(inputs, wiLocal, woLocal);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return no_diff eval(sd, wo, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        result = (BSDFSample)0;
        DustOnlyInputs inputs = sampleDustOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = dustOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        if (wiLocal.z < kMinCosTheta) return false;

        float diffusePdf;
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), diffusePdf);
        if (woLocal.z < kMinCosTheta) return false;

        result.wo = sf.fromLocal(woLocal);
        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo)) return false;
        result.pdf = dustOnlyPdf(woLocal);
        if (result.pdf <= 0.f) return false;
        result.weight = dustOnlyEval(inputs, wiLocal, woLocal) / result.pdf;
        result.lobeType = uint(LobeType::DiffuseReflection);
        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        DustOnlyInputs inputs = sampleDustOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = dustOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);
        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;
        return dustOnlyPdf(woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        DustOnlyInputs inputs = sampleDustOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = dustOnlyShadingFrame(data, sd, inputs);
        p.roughness = kDustOnlyRoughness;
        p.guideNormal = sf.N;
        p.diffuseReflectionAlbedo = inputs.coverage * kDustOnlyColor;
        return p;
    }

    uint getLobeTypes(const ShadingData sd) { return uint(LobeType::DiffuseReflection); }
};

struct ThreeLayeredGGXDustOnlyMaterial : MaterialBase, IMaterial, IBootstrapFeatureMaterial, ITrainingDirectionGuardMaterial
{
    ThreeLayeredGGXDustOnlyMaterialData data;
    typedef ThreeLayeredGGXDustOnlyMaterialInstance MaterialInstance;

    ThreeLayeredGGXDustOnlyMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        ThreeLayeredGGXDustOnlyMaterialInstance m = {};
        m.data = data;
        m.defaultSamplerID = header.getDefaultTextureSamplerID();
        return m;
    }

    [Differentiable]
    ThreeLayeredGGXDustOnlyMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = {};
        return setupMaterialInstance(ms, sd, lod, hints);
    }

    BootstrapFeatureData getBootstrapFeatures(const MaterialSystem ms, const ShadingData sd, const float3 wo)
    {
        BootstrapFeatureData features = {};
        uint samplerID = header.getDefaultTextureSamplerID();
        DustOnlyInputs inputs = sampleDustOnlyInputs(data, samplerID, sd);
        ShadingFrame sf = dustOnlyShadingFrame(data, sd, inputs);
        features.feature0 = float4(inputs.coverage, sf.N);
        features.feature1 = float4(sf.T, 0.f);
        return features;
    }

    bool isTrainingDirectionPairValid(const MaterialSystem ms, const ShadingData sd, const float3 wo, const float threshold)
    {
        uint samplerID = header.getDefaultTextureSamplerID();
        DustOnlyInputs inputs = sampleDustOnlyInputs(data, samplerID, sd);
        ShadingFrame sf = dustOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);
        if (min(wiLocal.z, woLocal.z) < max(threshold, kMinCosTheta)) return false;
        return isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo);
    }
};
