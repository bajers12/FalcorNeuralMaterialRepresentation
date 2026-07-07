import Scene.Scene;
import Scene.Material.BootstrapFeatureMaterial;
import Scene.Material.MaterialSystem;
import Scene.Material.TrainingDirectionGuardMaterial;
import Scene.Material.ShadingUtils;
import Rendering.Materials.AnisotropicGGX;
import Rendering.Materials.Fresnel;
import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import Utils.Math.ShadingFrame;

static const float kBaseOnlyPi = 3.14159265358979323846f;
static const float kBaseOnlyInvPi = 0.31830988618379067154f;

struct ThreeLayeredGGXBaseOnlyMaterialData
{
    TextureHandle texBaseColor = {};
    TextureHandle texNormal = {};
    TextureHandle texLayerRoughness = {};

    float baseF0 = 0.02f;
    float roughnessScale = 1.f;
    float roughnessBias = 0.02f;
    float normalFlatten = 0.f;

    uint flipNormalY = 1;
};

struct BaseOnlyInputs
{
    float3 baseColor;
    float roughness;
    float3 normalTS;
};

BaseOnlyInputs sampleBaseOnlyInputsAtUV(const ThreeLayeredGGXBaseOnlyMaterialData data, const uint samplerID, const float2 uv)
{
    SamplerState s = gScene.materials.getTextureSampler(samplerID);
    ExplicitLodTextureSampler lod = { 0.f };

    BaseOnlyInputs inputs;
    inputs.baseColor = gScene.materials.sampleTexture(data.texBaseColor, s, uv, float4(0.8f, 0.8f, 0.8f, 1.f), lod).rgb;

    float3 packedRoughness = gScene.materials.sampleTexture(
        data.texLayerRoughness,
        s,
        uv,
        float4(0.5f, 0.45f, 0.155f, 1.f),
        lod
    ).rgb;
    inputs.roughness = saturate(packedRoughness.r * data.roughnessScale + data.roughnessBias);

    inputs.normalTS = gScene.materials.sampleTexture(data.texNormal, s, uv, float4(0.5f, 0.5f, 1.f, 1.f), lod).xyz * 2.f - 1.f;
    if (data.flipNormalY != 0) inputs.normalTS.y = -inputs.normalTS.y;
    return inputs;
}

BaseOnlyInputs sampleBaseOnlyInputs(const ThreeLayeredGGXBaseOnlyMaterialData data, const uint samplerID, const ShadingData sd)
{
    return sampleBaseOnlyInputsAtUV(data, samplerID, sd.uv);
}

float baseOnlyRoughnessToAlpha(float roughness)
{
    float r = clamp(roughness, 0.02f, 1.f);
    return max(r * r, 1e-3f);
}

AnisotropicGGX baseOnlyDistribution(float roughness)
{
    float alpha = baseOnlyRoughnessToAlpha(roughness);
    return AnisotropicGGX(float2(alpha, alpha));
}

float3 baseOnlyFlattenNormal(float3 n, float flatten)
{
    n.xy *= 1.f - flatten;
    n.z = max(n.z, 1e-4f);
    return normalize(n);
}

ShadingFrame baseOnlyShadingFrame(const ThreeLayeredGGXBaseOnlyMaterialData data, const ShadingData sd, const BaseOnlyInputs inputs)
{
    float3 mapN = baseOnlyFlattenNormal(inputs.normalTS, saturate(data.normalFlatten));
    ShadingFrame sf = ShadingFrame(sd.frame.fromLocal(normalize(mapN)), sd.tangentW);
    flipShadingNormal(sd, sf);
    adjustShadingNormal(sd, sf);
    return sf;
}

float baseOnlyDiffusePdf(const float3 woLocal)
{
    return woLocal.z > kMinCosTheta ? woLocal.z * kBaseOnlyInvPi : 0.f;
}

float3 baseOnlyEvalSpecular(
    const ThreeLayeredGGXBaseOnlyMaterialData data,
    const BaseOnlyInputs inputs,
    const float3 wiLocal,
    const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f.xxx;

    float3 h = normalize(wiLocal + woLocal);
    float wiDotH = dot(wiLocal, h);
    if (h.z <= 0.f || wiDotH <= 0.f) return 0.f.xxx;

    AnisotropicGGX D = baseOnlyDistribution(inputs.roughness);
    float3 F = evalFresnelSchlick(saturate(data.baseF0).xxx, 1.f.xxx, saturate(wiDotH));
    return D.G2(woLocal, wiLocal) * D.evalNDF(h) * F / max(4.f * wiLocal.z, 1e-6f);
}

float baseOnlyEvalSpecularPdf(
    const ThreeLayeredGGXBaseOnlyMaterialData data,
    const BaseOnlyInputs inputs,
    const float3 wiLocal,
    const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;

    float3 h = normalize(wiLocal + woLocal);
    float wiDotH = dot(wiLocal, h);
    if (h.z <= 0.f || wiDotH <= 0.f) return 0.f;

    AnisotropicGGX D = baseOnlyDistribution(inputs.roughness);
    return D.evalPDF(wiLocal, h) / max(4.f * abs(wiDotH), 1e-6f);
}

void baseOnlyLobeWeights(const ThreeLayeredGGXBaseOnlyMaterialData data, const BaseOnlyInputs inputs, out float diffuseWeight, out float specularWeight)
{
    float specStrength = max(max(data.baseF0, data.baseF0), data.baseF0);
    float diffuseStrength = max(max(inputs.baseColor.r, inputs.baseColor.g), inputs.baseColor.b);
    float sum = max(diffuseStrength + specStrength, 1e-4f);
    diffuseWeight = diffuseStrength / sum;
    specularWeight = specStrength / sum;
}

float3 baseOnlyEval(
    const ThreeLayeredGGXBaseOnlyMaterialData data,
    const BaseOnlyInputs inputs,
    const float3 wiLocal,
    const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f.xxx;

    float3 diffuse = saturate(inputs.baseColor) * kBaseOnlyInvPi * woLocal.z;
    return diffuse + baseOnlyEvalSpecular(data, inputs, wiLocal, woLocal);
}

float baseOnlyPdf(
    const ThreeLayeredGGXBaseOnlyMaterialData data,
    const BaseOnlyInputs inputs,
    const float3 wiLocal,
    const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;

    float diffuseWeight;
    float specularWeight;
    baseOnlyLobeWeights(data, inputs, diffuseWeight, specularWeight);
    return diffuseWeight * baseOnlyDiffusePdf(woLocal) + specularWeight * baseOnlyEvalSpecularPdf(data, inputs, wiLocal, woLocal);
}

struct ThreeLayeredGGXBaseOnlyMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ThreeLayeredGGXBaseOnlyMaterialData data = {};
    uint defaultSamplerID = 0;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        BaseOnlyInputs inputs = sampleBaseOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = baseOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);
        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f.xxx;
        return baseOnlyEval(data, inputs, wiLocal, woLocal);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return no_diff eval(sd, wo, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        result = (BSDFSample)0;

        BaseOnlyInputs inputs = sampleBaseOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = baseOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        if (wiLocal.z < kMinCosTheta) return false;

        float diffuseWeight;
        float specularWeight;
        baseOnlyLobeWeights(data, inputs, diffuseWeight, specularWeight);

        float3 woLocal;
        if (sampleNext1D(sg) < diffuseWeight)
        {
            float diffusePdf;
            woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), diffusePdf);
            result.lobeType = uint(LobeType::DiffuseReflection);
        }
        else
        {
            AnisotropicGGX D = baseOnlyDistribution(inputs.roughness);
            float3 h = D.sample(wiLocal, sg);
            woLocal = reflect(-wiLocal, h);
            result.lobeType = uint(LobeType::SpecularReflection);
        }

        if (woLocal.z < kMinCosTheta) return false;

        float pdf = baseOnlyPdf(data, inputs, wiLocal, woLocal);
        if (pdf <= 0.f) return false;

        result.wo = sf.fromLocal(woLocal);
        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo)) return false;
        result.pdf = pdf;
        result.weight = baseOnlyEval(data, inputs, wiLocal, woLocal) / pdf;
        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        BaseOnlyInputs inputs = sampleBaseOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = baseOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);
        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;
        return baseOnlyPdf(data, inputs, wiLocal, woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.emission = 0.f.xxx;

        BaseOnlyInputs inputs = sampleBaseOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = baseOnlyShadingFrame(data, sd, inputs);
        p.roughness = inputs.roughness;
        p.guideNormal = sf.N;
        p.flags = 0;
        p.diffuseReflectionAlbedo = saturate(inputs.baseColor);
        p.diffuseTransmissionAlbedo = 0.f.xxx;
        p.specularReflectionAlbedo = saturate(data.baseF0).xxx;
        p.specularTransmissionAlbedo = 0.f.xxx;
        p.specularReflectance = p.specularReflectionAlbedo;
        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return uint(LobeType::DiffuseReflection) | uint(LobeType::SpecularReflection);
    }
};

struct ThreeLayeredGGXBaseOnlyMaterial : MaterialBase, IMaterial, IBootstrapFeatureMaterial, ITrainingDirectionGuardMaterial
{
    ThreeLayeredGGXBaseOnlyMaterialData data;

    typedef ThreeLayeredGGXBaseOnlyMaterialInstance MaterialInstance;

    ThreeLayeredGGXBaseOnlyMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        ThreeLayeredGGXBaseOnlyMaterialInstance m = {};
        m.data = data;
        m.defaultSamplerID = header.getDefaultTextureSamplerID();
        return m;
    }

    [Differentiable]
    ThreeLayeredGGXBaseOnlyMaterialInstance setupDiffMaterialInstance(
        out DiffMaterialData diffData,
        const MaterialSystem ms,
        const ShadingData sd,
        const ITextureSampler lod,
        const uint hints)
    {
        diffData = {};
        return setupMaterialInstance(ms, sd, lod, hints);
    }

    BootstrapFeatureData getBootstrapFeatures(const MaterialSystem ms, const ShadingData sd, const float3 wo)
    {
        BootstrapFeatureData features = {};

        uint samplerID = header.getDefaultTextureSamplerID();
        BaseOnlyInputs inputs = sampleBaseOnlyInputs(data, samplerID, sd);
        ShadingFrame sf = baseOnlyShadingFrame(data, sd, inputs);

        features.feature0 = float4(saturate(inputs.baseColor), inputs.roughness);
        features.feature1 = float4(sf.N, sf.T.x);
        features.feature2 = float4(sf.T.yz, 0.f, 0.f);
        return features;
    }

    bool isTrainingDirectionPairValid(const MaterialSystem ms, const ShadingData sd, const float3 wo, const float threshold)
    {
        uint samplerID = header.getDefaultTextureSamplerID();
        BaseOnlyInputs inputs = sampleBaseOnlyInputs(data, samplerID, sd);
        ShadingFrame sf = baseOnlyShadingFrame(data, sd, inputs);

        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);
        if (min(wiLocal.z, woLocal.z) < max(threshold, kMinCosTheta)) return false;
        return isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo);
    }
};
