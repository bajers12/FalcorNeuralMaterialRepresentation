import Scene.Scene;
import Scene.Material.MaterialSystem;
import Scene.Material.ShadingUtils;
import Rendering.Materials.AnisotropicGGX;
import Rendering.Materials.Fresnel;
import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import Utils.Math.ShadingFrame;

static const float kLayeredTexturePi = 3.14159265358979323846f;
static const float kLayeredTextureInvPi = 0.31830988618379067154f;
static const float kLayeredTextureHeightTexel = 1.f / 8192.f;

struct LayeredTextureMaterialData
{
    TextureHandle texBaseColor = {};
    TextureHandle texRoughness = {};
    TextureHandle texNormal = {};
    TextureHandle texDisplacement = {};
    TextureHandle texLayerRoughness = {};
    TextureHandle texLayerWeights = {};

    float baseF0 = 0.04f;
    float midF0 = 0.06f;
    float coatF0 = 0.08f;

    float roughnessScale = 1.f;
    float roughnessBias = 0.f;
    float thicknessScale = 1.f;
    float heightScale = 1.f;
    float3 absorptionColor = float3(0.35f, 0.25f, 0.15f);

    float baseWeightScale = 1.f;
    float midWeightScale = 1.f;
    float coatWeightScale = 1.f;

    float baseNormalFlatten = 0.f;
    float midNormalFlatten = 0.35f;
    float coatNormalFlatten = 0.7f;

    uint enableBaseLayer = 1;
    uint enableMidLayer = 1;
    uint enableCoatLayer = 1;
    uint flipNormalY = 1;
    uint showThickness = 0;
};

struct LayeredTextureInputs
{
    float3 baseColor;
    float baseRoughness;
    float midRoughness;
    float coatRoughness;
    float3 layerWeights;
    float3 normalTS;
    float displacement;
};

struct LayeredTextureLayerParams
{
    float roughness;
    float3 F0;
    float3 attenuation;
    float weight;
    float normalFlatten;
};

LayeredTextureInputs sampleLayeredTextureInputs(const LayeredTextureMaterialData data, const uint samplerID, const ShadingData sd)
{
    SamplerState s = gScene.materials.getTextureSampler(samplerID);
    ExplicitLodTextureSampler lod = { 0.f };

    LayeredTextureInputs inputs;
    inputs.baseColor = gScene.materials.sampleTexture(data.texBaseColor, s, sd.uv, float4(0.8f, 0.8f, 0.8f, 1.f), lod).rgb;

    float baseRoughnessFallback = gScene.materials.sampleTexture(data.texRoughness, s, sd.uv, float4(0.5f), lod).r;
    float3 packedRoughness = gScene.materials.sampleTexture(
        data.texLayerRoughness,
        s,
        sd.uv,
        float4(baseRoughnessFallback, saturate(baseRoughnessFallback * 0.6f + 0.15f), saturate(baseRoughnessFallback * 0.25f + 0.03f), 1.f),
        lod
    ).rgb;

    float cavity = saturate(gScene.materials.sampleTexture(data.texDisplacement, s, sd.uv, float4(0.f), lod).r);
    float3 packedWeights = gScene.materials.sampleTexture(
        data.texLayerWeights,
        s,
        sd.uv,
        float4(1.f - 0.35f * cavity, 0.35f + 0.45f * cavity, 0.2f + 0.2f * (1.f - cavity), 1.f),
        lod
    ).rgb;

    inputs.baseRoughness = saturate(packedRoughness.r * data.roughnessScale + data.roughnessBias);
    inputs.midRoughness = saturate(packedRoughness.g * data.roughnessScale + data.roughnessBias);
    inputs.coatRoughness = saturate(packedRoughness.b * data.roughnessScale + data.roughnessBias);

    inputs.layerWeights = max(0.f.xxx, packedWeights * float3(data.baseWeightScale, data.midWeightScale, data.coatWeightScale));

    inputs.normalTS = gScene.materials.sampleTexture(data.texNormal, s, sd.uv, float4(0.5f, 0.5f, 1.f, 1.f), lod).xyz * 2.f - 1.f;
    if (data.flipNormalY != 0) inputs.normalTS.y = -inputs.normalTS.y;
    inputs.displacement = cavity;
    return inputs;
}

float layeredTextureSampleDisplacement(const LayeredTextureMaterialData data, const uint samplerID, const float2 uv)
{
    SamplerState s = gScene.materials.getTextureSampler(samplerID);
    ExplicitLodTextureSampler lod = { 0.f };
    return gScene.materials.sampleTexture(data.texDisplacement, s, uv, float4(0.f), lod).r;
}

float layeredTextureRoughnessToAlpha(float roughness)
{
    float r = clamp(roughness, 0.02f, 1.f);
    return max(r * r, 1e-3f);
}

float layeredTextureThickness(const LayeredTextureMaterialData data, const LayeredTextureInputs inputs)
{
    return max(0.f, inputs.displacement * max(0.f, data.thicknessScale));
}

float3 layeredTextureAttenuation(const LayeredTextureMaterialData data, const LayeredTextureInputs inputs)
{
    float thickness = layeredTextureThickness(data, inputs);
    float3 absorption = max(data.absorptionColor, 0.f.xxx);
    return exp(-absorption * thickness);
}

float3 layeredTextureNormalizedLayerWeights(const LayeredTextureMaterialData data, const LayeredTextureInputs inputs)
{
    float3 weights = inputs.layerWeights;
    if (data.enableBaseLayer == 0) weights.x = 0.f;
    if (data.enableMidLayer == 0) weights.y = 0.f;
    if (data.enableCoatLayer == 0) weights.z = 0.f;

    float sum = weights.x + weights.y + weights.z;
    if (sum <= 1e-6f)
    {
        float3 fallback = float3(data.enableBaseLayer != 0 ? 1.f : 0.f, data.enableMidLayer != 0 ? 1.f : 0.f, data.enableCoatLayer != 0 ? 1.f : 0.f);
        sum = fallback.x + fallback.y + fallback.z;
        return sum > 0.f ? fallback / sum : 0.f.xxx;
    }

    return weights / sum;
}

LayeredTextureLayerParams layeredTextureGetBaseLayer(const LayeredTextureMaterialData data, const LayeredTextureInputs inputs)
{
    LayeredTextureLayerParams p;
    p.roughness = clamp(inputs.baseRoughness, 0.02f, 1.f);
    p.F0 = saturate(data.baseF0).xxx;
    float3 T = layeredTextureAttenuation(data, inputs);
    p.attenuation = T * T;
    p.weight = layeredTextureNormalizedLayerWeights(data, inputs).x;
    p.normalFlatten = saturate(data.baseNormalFlatten);
    return p;
}

LayeredTextureLayerParams layeredTextureGetMidLayer(const LayeredTextureMaterialData data, const LayeredTextureInputs inputs)
{
    LayeredTextureLayerParams p;
    p.roughness = clamp(inputs.midRoughness, 0.02f, 1.f);
    p.F0 = saturate(data.midF0).xxx;
    p.attenuation = layeredTextureAttenuation(data, inputs);
    p.weight = layeredTextureNormalizedLayerWeights(data, inputs).y;
    p.normalFlatten = saturate(data.midNormalFlatten);
    return p;
}

LayeredTextureLayerParams layeredTextureGetCoatLayer(const LayeredTextureMaterialData data, const LayeredTextureInputs inputs)
{
    LayeredTextureLayerParams p;
    p.roughness = clamp(inputs.coatRoughness, 0.02f, 1.f);
    p.F0 = saturate(data.coatF0).xxx;
    p.attenuation = 1.f.xxx;
    p.weight = layeredTextureNormalizedLayerWeights(data, inputs).z;
    p.normalFlatten = saturate(data.coatNormalFlatten);
    return p;
}

AnisotropicGGX layeredTextureDistribution(float roughness)
{
    float alpha = layeredTextureRoughnessToAlpha(roughness);
    return AnisotropicGGX(float2(alpha, alpha));
}

float3 layeredTextureFlattenNormal(float3 n, float flatten)
{
    n.xy *= 1.f - flatten;
    n.z = max(n.z, 1e-4f);
    return normalize(n);
}

ShadingFrame layeredTextureShadingFrame(const LayeredTextureMaterialData data, const uint samplerID, const ShadingData sd, const LayeredTextureInputs inputs, const float normalFlatten)
{
    float3 mapN = layeredTextureFlattenNormal(inputs.normalTS, normalFlatten);

    float2 du = float2(kLayeredTextureHeightTexel, 0.f);
    float2 dv = float2(0.f, kLayeredTextureHeightTexel);
    float hL = layeredTextureSampleDisplacement(data, samplerID, sd.uv - du);
    float hR = layeredTextureSampleDisplacement(data, samplerID, sd.uv + du);
    float hD = layeredTextureSampleDisplacement(data, samplerID, sd.uv - dv);
    float hU = layeredTextureSampleDisplacement(data, samplerID, sd.uv + dv);

    float heightScale = max(0.f, data.heightScale) * 0.01f;
    float flattenScale = 1.f - normalFlatten;
    float dHdu = (hR - hL) / (2.f * kLayeredTextureHeightTexel);
    float dHdv = (hU - hD) / (2.f * kLayeredTextureHeightTexel);

    mapN.xy += float2(-dHdu, -dHdv) * heightScale * flattenScale;
    mapN.z = max(mapN.z, 1e-4f);

    ShadingFrame sf = ShadingFrame(sd.frame.fromLocal(normalize(mapN)), sd.tangentW);
    adjustShadingNormal(sd, sf);
    return sf;
}

float layeredTextureDiffusePdf(const float3 woLocal)
{
    return woLocal.z > kMinCosTheta ? woLocal.z * kLayeredTextureInvPi : 0.f;
}

void layeredTextureBaseLobeWeights(const LayeredTextureInputs inputs, const LayeredTextureLayerParams layer, out float diffuseWeight, out float specularWeight)
{
    diffuseWeight = max(0.f, luminance(saturate(inputs.baseColor)));
    specularWeight = max(0.02f, luminance(layer.F0));
    float norm = 1.f / max(diffuseWeight + specularWeight, 1e-6f);
    diffuseWeight *= norm;
    specularWeight *= norm;
}

float3 layeredTextureEvalSpecular(const LayeredTextureLayerParams layer, const float3 wiLocal, const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f.xxx;

    float3 h = normalize(wiLocal + woLocal);
    float wiDotH = dot(wiLocal, h);
    if (h.z <= 0.f || wiDotH <= 0.f) return 0.f.xxx;

    AnisotropicGGX D = layeredTextureDistribution(layer.roughness);
    float3 F = evalFresnelSchlick(layer.F0, 1.f.xxx, wiDotH);
    return D.G2(woLocal, wiLocal) * D.evalNDF(h) * F / max(4.f * wiLocal.z, 1e-6f);
}

float layeredTextureEvalSpecularPdf(const LayeredTextureLayerParams layer, const float3 wiLocal, const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;

    float3 h = normalize(wiLocal + woLocal);
    float wiDotH = dot(wiLocal, h);
    if (h.z <= 0.f || wiDotH <= 0.f) return 0.f;

    AnisotropicGGX D = layeredTextureDistribution(layer.roughness);
    return D.evalPDF(wiLocal, h) / max(4.f * abs(wiDotH), 1e-6f);
}

float3 layeredTextureEvalBase(const LayeredTextureInputs inputs, const LayeredTextureLayerParams layer, const float3 wiLocal, const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f.xxx;

    float3 diffuse = saturate(inputs.baseColor) * kLayeredTextureInvPi * woLocal.z;
    return diffuse + layeredTextureEvalSpecular(layer, wiLocal, woLocal);
}

float layeredTextureEvalBasePdf(const LayeredTextureInputs inputs, const LayeredTextureLayerParams layer, const float3 wiLocal, const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;

    float diffuseWeight;
    float specularWeight;
    layeredTextureBaseLobeWeights(inputs, layer, diffuseWeight, specularWeight);
    return diffuseWeight * layeredTextureDiffusePdf(woLocal) + specularWeight * layeredTextureEvalSpecularPdf(layer, wiLocal, woLocal);
}

float3 layeredTextureEvalCombined(
    const LayeredTextureMaterialData data,
    const LayeredTextureInputs inputs,
    const float3 wiBaseLocal,
    const float3 woBaseLocal,
    const float3 wiMidLocal,
    const float3 woMidLocal,
    const float3 wiCoatLocal,
    const float3 woCoatLocal)
{
    LayeredTextureLayerParams baseLayer = layeredTextureGetBaseLayer(data, inputs);
    LayeredTextureLayerParams midLayer = layeredTextureGetMidLayer(data, inputs);
    LayeredTextureLayerParams coatLayer = layeredTextureGetCoatLayer(data, inputs);

    return baseLayer.weight * baseLayer.attenuation * layeredTextureEvalBase(inputs, baseLayer, wiBaseLocal, woBaseLocal)
         + midLayer.weight * midLayer.attenuation * layeredTextureEvalSpecular(midLayer, wiMidLocal, woMidLocal)
         + coatLayer.weight * layeredTextureEvalSpecular(coatLayer, wiCoatLocal, woCoatLocal);
}

float layeredTextureEvalCombinedPdf(
    const LayeredTextureMaterialData data,
    const LayeredTextureInputs inputs,
    const float3 wiBaseLocal,
    const float3 woBaseLocal,
    const float3 wiMidLocal,
    const float3 woMidLocal,
    const float3 wiCoatLocal,
    const float3 woCoatLocal)
{
    LayeredTextureLayerParams baseLayer = layeredTextureGetBaseLayer(data, inputs);
    LayeredTextureLayerParams midLayer = layeredTextureGetMidLayer(data, inputs);
    LayeredTextureLayerParams coatLayer = layeredTextureGetCoatLayer(data, inputs);

    return baseLayer.weight * layeredTextureEvalBasePdf(inputs, baseLayer, wiBaseLocal, woBaseLocal)
         + midLayer.weight * layeredTextureEvalSpecularPdf(midLayer, wiMidLocal, woMidLocal)
         + coatLayer.weight * layeredTextureEvalSpecularPdf(coatLayer, wiCoatLocal, woCoatLocal);
}

__exported import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Scene.Material.MaterialSystem;

struct LayeredTextureMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    LayeredTextureMaterialData data = {};
    uint defaultSamplerID = 0;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        LayeredTextureInputs inputs = sampleLayeredTextureInputs(data, defaultSamplerID, sd);
        if (data.showThickness != 0)
        {
            float thickness = saturate(layeredTextureThickness(data, inputs));
            return thickness.xxx * saturate(dot(sd.frame.N, wo));
        }

        LayeredTextureLayerParams baseLayer = layeredTextureGetBaseLayer(data, inputs);
        LayeredTextureLayerParams midLayer = layeredTextureGetMidLayer(data, inputs);
        LayeredTextureLayerParams coatLayer = layeredTextureGetCoatLayer(data, inputs);

        ShadingFrame baseSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, baseLayer.normalFlatten);
        ShadingFrame midSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, midLayer.normalFlatten);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, coatLayer.normalFlatten);

        float3 wiBaseLocal = baseSf.toLocal(sd.V);
        float3 woBaseLocal = baseSf.toLocal(wo);
        float3 wiMidLocal = midSf.toLocal(sd.V);
        float3 woMidLocal = midSf.toLocal(wo);
        float3 wiCoatLocal = coatSf.toLocal(sd.V);
        float3 woCoatLocal = coatSf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, baseSf, wiBaseLocal, woBaseLocal, wo)) return 0.f.xxx;
        if (!isValidHemisphereReflection(sd, midSf, wiMidLocal, woMidLocal, wo)) return 0.f.xxx;
        if (!isValidHemisphereReflection(sd, coatSf, wiCoatLocal, woCoatLocal, wo)) return 0.f.xxx;

        return layeredTextureEvalCombined(data, inputs, wiBaseLocal, woBaseLocal, wiMidLocal, woMidLocal, wiCoatLocal, woCoatLocal);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return no_diff eval(sd, wo, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        result = (BSDFSample)0;

        LayeredTextureInputs inputs = sampleLayeredTextureInputs(data, defaultSamplerID, sd);
        LayeredTextureLayerParams baseLayer = layeredTextureGetBaseLayer(data, inputs);
        LayeredTextureLayerParams midLayer = layeredTextureGetMidLayer(data, inputs);
        LayeredTextureLayerParams coatLayer = layeredTextureGetCoatLayer(data, inputs);

        ShadingFrame baseSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, baseLayer.normalFlatten);
        ShadingFrame midSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, midLayer.normalFlatten);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, coatLayer.normalFlatten);

        float3 wiBaseLocal = baseSf.toLocal(sd.V);
        float3 wiMidLocal = midSf.toLocal(sd.V);
        float3 wiCoatLocal = coatSf.toLocal(sd.V);
        if (wiBaseLocal.z < kMinCosTheta || wiMidLocal.z < kMinCosTheta || wiCoatLocal.z < kMinCosTheta) return false;

        if (data.showThickness != 0)
        {
            float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf);
            result.wo = baseSf.fromLocal(woLocal);
            if (!isValidHemisphereReflection(sd, baseSf, wiBaseLocal, woLocal, result.wo) || result.pdf <= 0.f) return false;

            float thickness = saturate(layeredTextureThickness(data, inputs));
            result.weight = thickness.xxx;
            result.lobeType = uint(LobeType::DiffuseReflection);
            return true;
        }

        float layerSample = sampleNext1D(sg);
        uint layerIndex = layerSample < coatLayer.weight ? 2u : (layerSample < coatLayer.weight + midLayer.weight ? 1u : 0u);

        float3 woWorld = 0.f.xxx;
        if (layerIndex == 2u)
        {
            AnisotropicGGX D = layeredTextureDistribution(coatLayer.roughness);
            float3 h = D.sample(wiCoatLocal, sg);
            float3 woLocal = reflect(-wiCoatLocal, h);
            if (woLocal.z < kMinCosTheta) return false;
            woWorld = coatSf.fromLocal(woLocal);
            result.lobeType = uint(LobeType::SpecularReflection);
        }
        else if (layerIndex == 1u)
        {
            AnisotropicGGX D = layeredTextureDistribution(midLayer.roughness);
            float3 h = D.sample(wiMidLocal, sg);
            float3 woLocal = reflect(-wiMidLocal, h);
            if (woLocal.z < kMinCosTheta) return false;
            woWorld = midSf.fromLocal(woLocal);
            result.lobeType = uint(LobeType::SpecularReflection);
        }
        else
        {
            float diffuseWeight;
            float specularWeight;
            layeredTextureBaseLobeWeights(inputs, baseLayer, diffuseWeight, specularWeight);

            float3 woLocal;
            if (sampleNext1D(sg) < diffuseWeight)
            {
                float diffusePdf;
                woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), diffusePdf);
            }
            else
            {
                AnisotropicGGX D = layeredTextureDistribution(baseLayer.roughness);
                float3 h = D.sample(wiBaseLocal, sg);
                woLocal = reflect(-wiBaseLocal, h);
            }

            if (woLocal.z < kMinCosTheta) return false;
            woWorld = baseSf.fromLocal(woLocal);
            result.lobeType = uint(LobeType::Reflection);
        }

        float3 woBaseLocal = baseSf.toLocal(woWorld);
        float3 woMidLocal = midSf.toLocal(woWorld);
        float3 woCoatLocal = coatSf.toLocal(woWorld);

        if (!isValidHemisphereReflection(sd, baseSf, wiBaseLocal, woBaseLocal, woWorld)) return false;
        if (!isValidHemisphereReflection(sd, midSf, wiMidLocal, woMidLocal, woWorld)) return false;
        if (!isValidHemisphereReflection(sd, coatSf, wiCoatLocal, woCoatLocal, woWorld)) return false;

        float pdf = layeredTextureEvalCombinedPdf(data, inputs, wiBaseLocal, woBaseLocal, wiMidLocal, woMidLocal, wiCoatLocal, woCoatLocal);
        if (pdf <= 0.f) return false;

        result.wo = woWorld;
        result.pdf = pdf;
        result.weight = layeredTextureEvalCombined(data, inputs, wiBaseLocal, woBaseLocal, wiMidLocal, woMidLocal, wiCoatLocal, woCoatLocal) / pdf;
        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        LayeredTextureInputs inputs = sampleLayeredTextureInputs(data, defaultSamplerID, sd);
        LayeredTextureLayerParams baseLayer = layeredTextureGetBaseLayer(data, inputs);
        LayeredTextureLayerParams midLayer = layeredTextureGetMidLayer(data, inputs);
        LayeredTextureLayerParams coatLayer = layeredTextureGetCoatLayer(data, inputs);

        ShadingFrame baseSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, baseLayer.normalFlatten);
        ShadingFrame midSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, midLayer.normalFlatten);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, coatLayer.normalFlatten);

        float3 wiBaseLocal = baseSf.toLocal(sd.V);
        float3 woBaseLocal = baseSf.toLocal(wo);
        float3 wiMidLocal = midSf.toLocal(sd.V);
        float3 woMidLocal = midSf.toLocal(wo);
        float3 wiCoatLocal = coatSf.toLocal(sd.V);
        float3 woCoatLocal = coatSf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, baseSf, wiBaseLocal, woBaseLocal, wo)) return 0.f;
        if (!isValidHemisphereReflection(sd, midSf, wiMidLocal, woMidLocal, wo)) return 0.f;
        if (!isValidHemisphereReflection(sd, coatSf, wiCoatLocal, woCoatLocal, wo)) return 0.f;
        if (data.showThickness != 0) return layeredTextureDiffusePdf(woBaseLocal);

        return layeredTextureEvalCombinedPdf(data, inputs, wiBaseLocal, woBaseLocal, wiMidLocal, woMidLocal, wiCoatLocal, woCoatLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.emission = 0.f.xxx;

        LayeredTextureInputs inputs = sampleLayeredTextureInputs(data, defaultSamplerID, sd);
        LayeredTextureLayerParams baseLayer = layeredTextureGetBaseLayer(data, inputs);
        LayeredTextureLayerParams midLayer = layeredTextureGetMidLayer(data, inputs);
        LayeredTextureLayerParams coatLayer = layeredTextureGetCoatLayer(data, inputs);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, coatLayer.normalFlatten);

        float totalWeight = max(baseLayer.weight + midLayer.weight + coatLayer.weight, 1e-6f);
        p.roughness =
            (baseLayer.weight * baseLayer.roughness + midLayer.weight * midLayer.roughness + coatLayer.weight * coatLayer.roughness) /
            totalWeight;
        p.guideNormal = coatSf.N;
        p.flags = 0;

        p.diffuseReflectionAlbedo = baseLayer.weight * baseLayer.attenuation * saturate(inputs.baseColor);
        p.diffuseTransmissionAlbedo = 0.f.xxx;
        p.specularReflectionAlbedo =
            (baseLayer.weight * baseLayer.attenuation * baseLayer.F0 + midLayer.weight * midLayer.attenuation * midLayer.F0 + coatLayer.weight * coatLayer.F0) /
            totalWeight;
        p.specularTransmissionAlbedo = 0.f.xxx;
        p.specularReflectance = p.specularReflectionAlbedo;
        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return uint(LobeType::DiffuseReflection) | uint(LobeType::SpecularReflection);
    }
};

struct LayeredTextureMaterial : MaterialBase, IMaterial
{
    LayeredTextureMaterialData data;

    typedef LayeredTextureMaterialInstance MaterialInstance;

    LayeredTextureMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        LayeredTextureMaterialInstance m = {};
        m.data = data;
        m.defaultSamplerID = header.getDefaultTextureSamplerID();
        return m;
    }

    [Differentiable]
    LayeredTextureMaterialInstance setupDiffMaterialInstance(
        out DiffMaterialData diffData,
        const MaterialSystem ms,
        const ShadingData sd,
        const ITextureSampler lod,
        const uint hints)
    {
        diffData = {};
        return setupMaterialInstance(ms, sd, lod, hints);
    }
};
