import Scene.Scene;
import Scene.Material.BootstrapFeatureMaterial;
import Scene.Material.MaterialSystem;
import Scene.Material.TrainingDirectionGuardMaterial;
import Scene.Material.ShadingUtils;
import Rendering.Materials.AnisotropicGGX;
import Rendering.Materials.Fresnel;
import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Utils.Math.ShadingFrame;

struct ThreeLayeredGGXMidOnlyMaterialData
{
    TextureHandle texBaseColor = {};
    TextureHandle texNormal = {};
    TextureHandle texLayerRoughness = {};
    TextureHandle texDustCoverage = {};

    float f0 = 0.2f;
    float roughnessScale = 1.f;
    float roughnessBias = 0.02f;
    float normalFlatten = 0.2f;

    uint flipNormalY = 1;
};

struct MidOnlyInputs
{
    float roughness;
    float3 normalTS;
};

MidOnlyInputs sampleMidOnlyInputsAtUV(const ThreeLayeredGGXMidOnlyMaterialData data, const uint samplerID, const float2 uv)
{
    SamplerState s = gScene.materials.getTextureSampler(samplerID);
    ExplicitLodTextureSampler lod = { 0.f };

    float3 packedRoughness = gScene.materials.sampleTexture(data.texLayerRoughness, s, uv, float4(0.5f, 0.45f, 0.155f, 1.f), lod).rgb;
    MidOnlyInputs inputs;
    inputs.roughness = saturate(packedRoughness.g * data.roughnessScale + data.roughnessBias);
    inputs.normalTS = gScene.materials.sampleTexture(data.texNormal, s, uv, float4(0.5f, 0.5f, 1.f, 1.f), lod).xyz * 2.f - 1.f;
    if (data.flipNormalY != 0) inputs.normalTS.y = -inputs.normalTS.y;
    return inputs;
}

MidOnlyInputs sampleMidOnlyInputs(const ThreeLayeredGGXMidOnlyMaterialData data, const uint samplerID, const ShadingData sd)
{
    return sampleMidOnlyInputsAtUV(data, samplerID, sd.uv);
}

float3 midOnlyFlattenNormal(float3 n, float flatten)
{
    n.xy *= 1.f - flatten;
    n.z = max(n.z, 1e-4f);
    return normalize(n);
}

ShadingFrame midOnlyShadingFrame(const ThreeLayeredGGXMidOnlyMaterialData data, const ShadingData sd, const MidOnlyInputs inputs)
{
    float3 mapN = midOnlyFlattenNormal(inputs.normalTS, saturate(data.normalFlatten));
    ShadingFrame sf = ShadingFrame(sd.frame.fromLocal(normalize(mapN)), sd.tangentW);
    flipShadingNormal(sd, sf);
    adjustShadingNormal(sd, sf);
    return sf;
}

AnisotropicGGX midOnlyDistribution(float roughness)
{
    float r = clamp(roughness, 0.02f, 1.f);
    float alpha = max(r * r, 1e-3f);
    return AnisotropicGGX(float2(alpha, alpha));
}

float3 midOnlyEval(const ThreeLayeredGGXMidOnlyMaterialData data, const MidOnlyInputs inputs, const float3 wiLocal, const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f.xxx;

    float3 h = normalize(wiLocal + woLocal);
    float wiDotH = dot(wiLocal, h);
    if (h.z <= 0.f || wiDotH <= 0.f) return 0.f.xxx;

    AnisotropicGGX D = midOnlyDistribution(inputs.roughness);
    float3 F = evalFresnelSchlick(saturate(data.f0).xxx, 1.f.xxx, saturate(wiDotH));
    return D.G2(woLocal, wiLocal) * D.evalNDF(h) * F / max(4.f * wiLocal.z, 1e-6f);
}

float midOnlyPdf(const ThreeLayeredGGXMidOnlyMaterialData data, const MidOnlyInputs inputs, const float3 wiLocal, const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;

    float3 h = normalize(wiLocal + woLocal);
    float wiDotH = dot(wiLocal, h);
    if (h.z <= 0.f || wiDotH <= 0.f) return 0.f;

    AnisotropicGGX D = midOnlyDistribution(inputs.roughness);
    return D.evalPDF(wiLocal, h) / max(4.f * abs(wiDotH), 1e-6f);
}

struct ThreeLayeredGGXMidOnlyMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ThreeLayeredGGXMidOnlyMaterialData data = {};
    uint defaultSamplerID = 0;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        MidOnlyInputs inputs = sampleMidOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = midOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);
        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f.xxx;
        return midOnlyEval(data, inputs, wiLocal, woLocal);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return no_diff eval(sd, wo, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        result = (BSDFSample)0;
        MidOnlyInputs inputs = sampleMidOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = midOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        if (wiLocal.z < kMinCosTheta) return false;

        AnisotropicGGX D = midOnlyDistribution(inputs.roughness);
        float3 h = D.sample(wiLocal, sg);
        float3 woLocal = reflect(-wiLocal, h);
        if (woLocal.z < kMinCosTheta) return false;

        result.wo = sf.fromLocal(woLocal);
        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo)) return false;
        result.pdf = midOnlyPdf(data, inputs, wiLocal, woLocal);
        if (result.pdf <= 0.f) return false;
        result.weight = midOnlyEval(data, inputs, wiLocal, woLocal) / result.pdf;
        result.lobeType = uint(LobeType::SpecularReflection);
        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        MidOnlyInputs inputs = sampleMidOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = midOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);
        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;
        return midOnlyPdf(data, inputs, wiLocal, woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        MidOnlyInputs inputs = sampleMidOnlyInputs(data, defaultSamplerID, sd);
        ShadingFrame sf = midOnlyShadingFrame(data, sd, inputs);
        p.roughness = inputs.roughness;
        p.guideNormal = sf.N;
        p.specularReflectionAlbedo = saturate(data.f0).xxx;
        p.specularReflectance = p.specularReflectionAlbedo;
        return p;
    }

    uint getLobeTypes(const ShadingData sd) { return uint(LobeType::SpecularReflection); }
};

struct ThreeLayeredGGXMidOnlyMaterial : MaterialBase, IMaterial, IBootstrapFeatureMaterial, ITrainingDirectionGuardMaterial
{
    ThreeLayeredGGXMidOnlyMaterialData data;
    typedef ThreeLayeredGGXMidOnlyMaterialInstance MaterialInstance;

    ThreeLayeredGGXMidOnlyMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        ThreeLayeredGGXMidOnlyMaterialInstance m = {};
        m.data = data;
        m.defaultSamplerID = header.getDefaultTextureSamplerID();
        return m;
    }

    [Differentiable]
    ThreeLayeredGGXMidOnlyMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = {};
        return setupMaterialInstance(ms, sd, lod, hints);
    }

    BootstrapFeatureData getBootstrapFeatures(const MaterialSystem ms, const ShadingData sd, const float3 wo)
    {
        BootstrapFeatureData features = {};
        uint samplerID = header.getDefaultTextureSamplerID();
        MidOnlyInputs inputs = sampleMidOnlyInputs(data, samplerID, sd);
        ShadingFrame sf = midOnlyShadingFrame(data, sd, inputs);
        features.feature0 = float4(inputs.roughness, sf.N);
        features.feature1 = float4(sf.T, 0.f);
        return features;
    }

    bool isTrainingDirectionPairValid(const MaterialSystem ms, const ShadingData sd, const float3 wo, const float threshold)
    {
        uint samplerID = header.getDefaultTextureSamplerID();
        MidOnlyInputs inputs = sampleMidOnlyInputs(data, samplerID, sd);
        ShadingFrame sf = midOnlyShadingFrame(data, sd, inputs);
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);
        if (min(wiLocal.z, woLocal.z) < max(threshold, kMinCosTheta)) return false;
        return isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo);
    }
};
