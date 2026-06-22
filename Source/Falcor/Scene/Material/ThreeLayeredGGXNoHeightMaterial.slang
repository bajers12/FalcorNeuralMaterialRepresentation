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

static const float kLayeredTexturePi = 3.14159265358979323846f;
static const float kLayeredTextureInvPi = 0.31830988618379067154f;
static const float3 kDustColor = float3(0.55f, 0.52f, 0.46f);
static const float kDustRoughness = 0.8f;
static const float3 kMidAbsorptionCoefficient = float3(0.25f, 0.2f, 0.15f);
static const float kMidOpticalThickness = 0.5f;
static const uint kMicrofacetSampleCount = 64u;

struct ThreeLayeredGGXNoHeightMaterialData
{
    TextureHandle texBaseColor = {};
    TextureHandle texNormal = {};
    TextureHandle texLayerRoughness = {};
    TextureHandle texDustCoverage = {};

    float baseF0 = 0.02f;
    float midF0 = 0.2f;
    float coatF0 = 0.3f;

    float roughnessScale = 1.f;
    float roughnessBias = 0.02f;

    float dustCoverageScale = 0.65f;

    float baseNormalFlatten = 0.f;
    float midNormalFlatten = 0.2f;
    float coatNormalFlatten = 0.8f;

    uint enableBaseLayer = 1;
    uint enableMidLayer = 1;
    uint enableCoatLayer = 1;
    uint flipNormalY = 1;
};

struct LayeredTextureInputs
{
    float3 baseColor;
    float baseRoughness;
    float midRoughness;
    float coatRoughness;
    float dustCoverage;
    float3 normalTS;
};

struct LayeredTextureLayerParams
{
    float roughness;
    float3 F0;
    float coverage;
    float normalFlatten;
};

struct LayeredTextureDustParams
{
    float3 color;
    float roughness;
    float coverage;
};

LayeredTextureInputs sampleLayeredTextureInputsAtUV(const ThreeLayeredGGXNoHeightMaterialData data, const uint samplerID, const float2 uv)
{
    SamplerState s = gScene.materials.getTextureSampler(samplerID);
    ExplicitLodTextureSampler lod = { 0.f };

    LayeredTextureInputs inputs;
    inputs.baseColor = gScene.materials.sampleTexture(data.texBaseColor, s, uv, float4(0.8f, 0.8f, 0.8f, 1.f), lod).rgb;

    float3 packedRoughness = gScene.materials.sampleTexture(
        data.texLayerRoughness,
        s,
        uv,
        float4(0.5f, 0.45f, 0.155f, 1.f),
        lod
    ).rgb;

    inputs.baseRoughness = saturate(packedRoughness.r * data.roughnessScale + data.roughnessBias);
    inputs.midRoughness = saturate(packedRoughness.g * data.roughnessScale + data.roughnessBias);
    inputs.coatRoughness = saturate(packedRoughness.b * data.roughnessScale + data.roughnessBias);

    float dustCoverage = gScene.materials.sampleTexture(data.texDustCoverage, s, uv, float4(0.f), lod).r;
    inputs.dustCoverage = saturate(dustCoverage * data.dustCoverageScale);

    inputs.normalTS = gScene.materials.sampleTexture(data.texNormal, s, uv, float4(0.5f, 0.5f, 1.f, 1.f), lod).xyz * 2.f - 1.f;
    if (data.flipNormalY != 0) inputs.normalTS.y = -inputs.normalTS.y;
    return inputs;
}

LayeredTextureInputs sampleLayeredTextureInputs(const ThreeLayeredGGXNoHeightMaterialData data, const uint samplerID, const ShadingData sd)
{
    return sampleLayeredTextureInputsAtUV(data, samplerID, sd.uv);
}

float layeredTextureRoughnessToAlpha(float roughness)
{
    float r = clamp(roughness, 0.02f, 1.f);
    return max(r * r, 1e-3f);
}

float4 layeredTextureLayerControls(const ThreeLayeredGGXNoHeightMaterialData data, const LayeredTextureInputs inputs)
{
    float4 controls = float4(1.f, 1.f, 1.f, saturate(inputs.dustCoverage));
    if (data.enableBaseLayer == 0) controls.x = 0.f;
    if (data.enableMidLayer == 0) controls.y = 0.f;
    if (data.enableCoatLayer == 0) controls.z = 0.f;
    return controls;
}

LayeredTextureLayerParams layeredTextureGetBaseLayer(const ThreeLayeredGGXNoHeightMaterialData data, const LayeredTextureInputs inputs)
{
    LayeredTextureLayerParams p;
    p.roughness = clamp(inputs.baseRoughness, 0.02f, 1.f);
    p.F0 = saturate(data.baseF0).xxx;
    p.coverage = layeredTextureLayerControls(data, inputs).x;
    p.normalFlatten = saturate(data.baseNormalFlatten);
    return p;
}

LayeredTextureLayerParams layeredTextureGetMidLayer(const ThreeLayeredGGXNoHeightMaterialData data, const LayeredTextureInputs inputs)
{
    LayeredTextureLayerParams p;
    p.roughness = clamp(inputs.midRoughness, 0.02f, 1.f);
    p.F0 = saturate(data.midF0).xxx;
    p.coverage = layeredTextureLayerControls(data, inputs).y;
    p.normalFlatten = saturate(data.midNormalFlatten);
    return p;
}

LayeredTextureLayerParams layeredTextureGetCoatLayer(const ThreeLayeredGGXNoHeightMaterialData data, const LayeredTextureInputs inputs)
{
    LayeredTextureLayerParams p;
    p.roughness = clamp(inputs.coatRoughness, 0.02f, 1.f);
    p.F0 = saturate(data.coatF0).xxx;
    p.coverage = layeredTextureLayerControls(data, inputs).z;
    p.normalFlatten = saturate(data.coatNormalFlatten);
    return p;
}

LayeredTextureDustParams layeredTextureGetDustLayer(const ThreeLayeredGGXNoHeightMaterialData data, const LayeredTextureInputs inputs)
{
    LayeredTextureDustParams p;
    p.color = kDustColor;
    p.roughness = kDustRoughness;
    p.coverage = layeredTextureLayerControls(data, inputs).w;
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

ShadingFrame layeredTextureShadingFrameAtUV(const ThreeLayeredGGXNoHeightMaterialData data, const uint samplerID, const ShadingData sd, const LayeredTextureInputs inputs, const float normalFlatten, const float2 uv)
{
    float3 mapN = layeredTextureFlattenNormal(inputs.normalTS, normalFlatten);

    ShadingFrame sf = ShadingFrame(sd.frame.fromLocal(normalize(mapN)), sd.tangentW);
    flipShadingNormal(sd, sf);
    adjustShadingNormal(sd, sf);
    return sf;
}

ShadingFrame layeredTextureShadingFrame(const ThreeLayeredGGXNoHeightMaterialData data, const uint samplerID, const ShadingData sd, const LayeredTextureInputs inputs, const float normalFlatten)
{
    return layeredTextureShadingFrameAtUV(data, samplerID, sd, inputs, normalFlatten, sd.uv);
}

float layeredTextureDiffusePdf(const float3 woLocal)
{
    return woLocal.z > kMinCosTheta ? woLocal.z * kLayeredTextureInvPi : 0.f;
}

float3 layeredTextureDirectionalFresnel(const LayeredTextureLayerParams layer, const float3 directionLocal)
{
    return evalFresnelSchlick(layer.F0, 1.f.xxx, saturate(abs(directionLocal.z)));
}

float3 layeredTextureTwoWayTransmission(
    const LayeredTextureLayerParams layer,
    const float3 wiLocal,
    const float3 woLocal)
{
    return (1.f.xxx - layeredTextureDirectionalFresnel(layer, wiLocal)) *
           (1.f.xxx - layeredTextureDirectionalFresnel(layer, woLocal));
}

float3 layeredTextureBeerTransmission(
    const float3 absorptionCoefficient,
    const float thickness,
    const float3 wiLocal,
    const float3 woLocal)
{
    float pathLength =
        max(thickness, 0.f) *
        (rcp(max(abs(wiLocal.z), 0.1f)) + rcp(max(abs(woLocal.z), 0.1f)));
    return exp(-max(absorptionCoefficient, 0.f.xxx) * pathLength);
}

float3 layeredTextureMidTransmission(
    const LayeredTextureLayerParams midLayer,
    const float3 wiLocal,
    const float3 woLocal)
{
    return layeredTextureTwoWayTransmission(midLayer, wiLocal, woLocal) *
           layeredTextureBeerTransmission(kMidAbsorptionCoefficient, kMidOpticalThickness, wiLocal, woLocal);
}

float4 layeredTextureSamplingWeights(
    const LayeredTextureInputs inputs,
    const LayeredTextureLayerParams baseLayer,
    const LayeredTextureLayerParams midLayer,
    const LayeredTextureLayerParams coatLayer,
    const LayeredTextureDustParams dustLayer,
    const float3 wiMidLocal,
    const float3 wiCoatLocal)
{
    float dustProbability = dustLayer.coverage;
    float visibleBelowDust = 1.f - dustLayer.coverage;

    float coatFresnel = luminance(layeredTextureDirectionalFresnel(coatLayer, wiCoatLocal));
    float coatTransmission = lerp(1.f, 1.f - coatFresnel, coatLayer.coverage);
    float coatProbability = visibleBelowDust * coatLayer.coverage * max(coatFresnel, 0.02f);

    float midFresnel = luminance(layeredTextureDirectionalFresnel(midLayer, wiMidLocal));
    float3 midBeer = exp(
        -kMidAbsorptionCoefficient *
        (kMidOpticalThickness * rcp(max(abs(wiMidLocal.z), 0.1f)))
    );
    float midTransmission = lerp(1.f, (1.f - midFresnel) * luminance(midBeer), midLayer.coverage);
    float midProbability = visibleBelowDust * coatTransmission * midLayer.coverage * max(midFresnel, 0.02f);

    float baseImportance = max(luminance(saturate(inputs.baseColor)), luminance(baseLayer.F0));
    float baseProbability = visibleBelowDust * coatTransmission * midTransmission * baseLayer.coverage * max(baseImportance, 0.02f);

    float4 probabilities = float4(baseProbability, midProbability, coatProbability, dustProbability);
    float sum = probabilities.x + probabilities.y + probabilities.z + probabilities.w;
    return sum > 1e-6f ? probabilities / sum : float4(1.f, 0.f, 0.f, 0.f);
}

float3 layeredTextureEvalOrenNayar(const LayeredTextureDustParams layer, const float3 wiLocal, const float3 woLocal)
{
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f.xxx;

    float sinThetaI = sqrt(max(0.f, 1.f - wiLocal.z * wiLocal.z));
    float sinThetaO = sqrt(max(0.f, 1.f - woLocal.z * woLocal.z));
    float sinPhiI = sinThetaI == 0.f ? 0.f : clamp(wiLocal.y / sinThetaI, -1.f, 1.f);
    float cosPhiI = sinThetaI == 0.f ? 0.f : clamp(wiLocal.x / sinThetaI, -1.f, 1.f);
    float sinPhiO = sinThetaO == 0.f ? 0.f : clamp(woLocal.y / sinThetaO, -1.f, 1.f);
    float cosPhiO = sinThetaO == 0.f ? 0.f : clamp(woLocal.x / sinThetaO, -1.f, 1.f);
    float cosDiff = max(cosPhiI * cosPhiO + sinPhiI * sinPhiO, 0.f);

    float sigma2 = layer.roughness * layer.roughness;
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

    return layer.color * kLayeredTextureInvPi * (A + B * cosDiff * sinAlpha * tanBeta) * woLocal.z;
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

float3 layeredTextureEvalCombinedSingle(
    const ThreeLayeredGGXNoHeightMaterialData data,
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
    LayeredTextureDustParams dustLayer = layeredTextureGetDustLayer(data, inputs);

    float3 midTransmission = lerp(
        1.f.xxx,
        layeredTextureMidTransmission(midLayer, wiMidLocal, woMidLocal),
        midLayer.coverage
    );
    float3 coatTransmission = lerp(
        1.f.xxx,
        layeredTextureTwoWayTransmission(coatLayer, wiCoatLocal, woCoatLocal),
        coatLayer.coverage
    );

    float3 baseContribution = baseLayer.coverage * layeredTextureEvalBase(inputs, baseLayer, wiBaseLocal, woBaseLocal);
    float3 midStack =
        midLayer.coverage * layeredTextureEvalSpecular(midLayer, wiMidLocal, woMidLocal) +
        midTransmission * baseContribution;
    float3 dielectricStack =
        coatLayer.coverage * layeredTextureEvalSpecular(coatLayer, wiCoatLocal, woCoatLocal) +
        coatTransmission * midStack;
    return dustLayer.coverage * layeredTextureEvalOrenNayar(dustLayer, wiCoatLocal, woCoatLocal) +
           (1.f - dustLayer.coverage) * dielectricStack;
}

float layeredTextureEvalCombinedPdfSingle(
    const ThreeLayeredGGXNoHeightMaterialData data,
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
    LayeredTextureDustParams dustLayer = layeredTextureGetDustLayer(data, inputs);
    float4 samplingWeights = layeredTextureSamplingWeights(inputs, baseLayer, midLayer, coatLayer, dustLayer, wiMidLocal, wiCoatLocal);

    return samplingWeights.x * layeredTextureEvalBasePdf(inputs, baseLayer, wiBaseLocal, woBaseLocal)
         + samplingWeights.y * layeredTextureEvalSpecularPdf(midLayer, wiMidLocal, woMidLocal)
         + samplingWeights.z * layeredTextureEvalSpecularPdf(coatLayer, wiCoatLocal, woCoatLocal)
         + samplingWeights.w * layeredTextureDiffusePdf(woCoatLocal);
}

uint layeredTextureMicrofacetSampleCount(const ThreeLayeredGGXNoHeightMaterialData data)
{
    return kMicrofacetSampleCount;
}

float layeredTextureMicrofacetVariant(const uint sampleIndex, const uint channel)
{
    float x = float(sampleIndex) * 12.9898f + float(channel) * 78.233f;
    return frac(sin(x) * 43758.5453f);
}

LayeredTextureLayerParams layeredTextureMicrofacetLayerVariant(LayeredTextureLayerParams layer, const uint sampleIndex, const uint sampleCount, const uint layerIndex)
{
    if (sampleIndex == 0u) return layer;

    float u = layeredTextureMicrofacetVariant(sampleIndex, layerIndex * 3u + 0u);
    float f0Jitter = layeredTextureMicrofacetVariant(sampleIndex, layerIndex * 3u + 1u) * 2.f - 1.f;
    float strength = saturate((float(sampleCount) - 1.f) / 15.f);
    float tailWeight = pow(u, 2.5f);
    float sharpRoughness = max(0.02f, layer.roughness * 0.35f);
    float broadRoughness = min(1.f, sqrt(max(layer.roughness, 0.02f)));
    float variantRoughness = lerp(sharpRoughness, broadRoughness, tailWeight);

    layer.roughness = lerp(layer.roughness, variantRoughness, strength);
    layer.F0 = saturate(layer.F0 * (1.f + strength * 0.25f * f0Jitter));
    return layer;
}

float3 layeredTextureEvalCombined(
    const ThreeLayeredGGXNoHeightMaterialData data,
    const uint samplerID,
    const ShadingData sd,
    const LayeredTextureInputs centerInputs,
    const float3 wo)
{
    uint sampleCount = layeredTextureMicrofacetSampleCount(data);
    float3 accum = 0.f.xxx;
    float validCount = 0.f;

    [loop]
    for (uint i = 0u; i < sampleCount; ++i)
    {
        LayeredTextureInputs inputs = centerInputs;

        LayeredTextureLayerParams baseLayer = layeredTextureMicrofacetLayerVariant(layeredTextureGetBaseLayer(data, inputs), i, sampleCount, 0u);
        LayeredTextureLayerParams midLayer = layeredTextureMicrofacetLayerVariant(layeredTextureGetMidLayer(data, inputs), i, sampleCount, 1u);
        LayeredTextureLayerParams coatLayer = layeredTextureMicrofacetLayerVariant(layeredTextureGetCoatLayer(data, inputs), i, sampleCount, 2u);
        LayeredTextureDustParams dustLayer = layeredTextureGetDustLayer(data, inputs);

        ShadingFrame baseSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, baseLayer.normalFlatten);
        ShadingFrame midSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, midLayer.normalFlatten);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, coatLayer.normalFlatten);

        float3 wiBaseLocal = baseSf.toLocal(sd.V);
        float3 woBaseLocal = baseSf.toLocal(wo);
        float3 wiMidLocal = midSf.toLocal(sd.V);
        float3 woMidLocal = midSf.toLocal(wo);
        float3 wiCoatLocal = coatSf.toLocal(sd.V);
        float3 woCoatLocal = coatSf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, baseSf, wiBaseLocal, woBaseLocal, wo)) continue;
        if (!isValidHemisphereReflection(sd, midSf, wiMidLocal, woMidLocal, wo)) continue;
        if (!isValidHemisphereReflection(sd, coatSf, wiCoatLocal, woCoatLocal, wo)) continue;

        float3 midTransmission = lerp(
            1.f.xxx,
            layeredTextureMidTransmission(midLayer, wiMidLocal, woMidLocal),
            midLayer.coverage
        );
        float3 coatTransmission = lerp(
            1.f.xxx,
            layeredTextureTwoWayTransmission(coatLayer, wiCoatLocal, woCoatLocal),
            coatLayer.coverage
        );
        float3 baseContribution = baseLayer.coverage * layeredTextureEvalBase(inputs, baseLayer, wiBaseLocal, woBaseLocal);
        float3 midStack =
            midLayer.coverage * layeredTextureEvalSpecular(midLayer, wiMidLocal, woMidLocal) +
            midTransmission * baseContribution;
        float3 dielectricStack =
            coatLayer.coverage * layeredTextureEvalSpecular(coatLayer, wiCoatLocal, woCoatLocal) +
            coatTransmission * midStack;
        accum += dustLayer.coverage * layeredTextureEvalOrenNayar(dustLayer, wiCoatLocal, woCoatLocal) +
            (1.f - dustLayer.coverage) * dielectricStack;
        validCount += 1.f;
    }

    return validCount > 0.f ? accum / validCount : 0.f.xxx;
}

float layeredTextureEvalCombinedPdf(
    const ThreeLayeredGGXNoHeightMaterialData data,
    const uint samplerID,
    const ShadingData sd,
    const LayeredTextureInputs centerInputs,
    const float3 wo)
{
    uint sampleCount = layeredTextureMicrofacetSampleCount(data);
    float accum = 0.f;
    float validCount = 0.f;

    [loop]
    for (uint i = 0u; i < sampleCount; ++i)
    {
        LayeredTextureInputs inputs = centerInputs;

        LayeredTextureLayerParams baseLayer = layeredTextureMicrofacetLayerVariant(layeredTextureGetBaseLayer(data, inputs), i, sampleCount, 0u);
        LayeredTextureLayerParams midLayer = layeredTextureMicrofacetLayerVariant(layeredTextureGetMidLayer(data, inputs), i, sampleCount, 1u);
        LayeredTextureLayerParams coatLayer = layeredTextureMicrofacetLayerVariant(layeredTextureGetCoatLayer(data, inputs), i, sampleCount, 2u);
        LayeredTextureDustParams dustLayer = layeredTextureGetDustLayer(data, inputs);

        ShadingFrame baseSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, baseLayer.normalFlatten);
        ShadingFrame midSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, midLayer.normalFlatten);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, coatLayer.normalFlatten);

        float3 wiBaseLocal = baseSf.toLocal(sd.V);
        float3 woBaseLocal = baseSf.toLocal(wo);
        float3 wiMidLocal = midSf.toLocal(sd.V);
        float3 woMidLocal = midSf.toLocal(wo);
        float3 wiCoatLocal = coatSf.toLocal(sd.V);
        float3 woCoatLocal = coatSf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, baseSf, wiBaseLocal, woBaseLocal, wo)) continue;
        if (!isValidHemisphereReflection(sd, midSf, wiMidLocal, woMidLocal, wo)) continue;
        if (!isValidHemisphereReflection(sd, coatSf, wiCoatLocal, woCoatLocal, wo)) continue;

        float4 samplingWeights = layeredTextureSamplingWeights(inputs, baseLayer, midLayer, coatLayer, dustLayer, wiMidLocal, wiCoatLocal);
        accum += samplingWeights.x * layeredTextureEvalBasePdf(inputs, baseLayer, wiBaseLocal, woBaseLocal)
            + samplingWeights.y * layeredTextureEvalSpecularPdf(midLayer, wiMidLocal, woMidLocal)
            + samplingWeights.z * layeredTextureEvalSpecularPdf(coatLayer, wiCoatLocal, woCoatLocal)
            + samplingWeights.w * layeredTextureDiffusePdf(woCoatLocal);
        validCount += 1.f;
    }

    return validCount > 0.f ? accum / validCount : 0.f;
}

__exported import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Scene.Material.MaterialSystem;

struct ThreeLayeredGGXNoHeightMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ThreeLayeredGGXNoHeightMaterialData data = {};
    uint defaultSamplerID = 0;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        LayeredTextureInputs inputs = sampleLayeredTextureInputs(data, defaultSamplerID, sd);
        return layeredTextureEvalCombined(data, defaultSamplerID, sd, inputs, wo);
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
        LayeredTextureDustParams dustLayer = layeredTextureGetDustLayer(data, inputs);

        ShadingFrame baseSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, baseLayer.normalFlatten);
        ShadingFrame midSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, midLayer.normalFlatten);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, coatLayer.normalFlatten);

        float3 wiBaseLocal = baseSf.toLocal(sd.V);
        float3 wiMidLocal = midSf.toLocal(sd.V);
        float3 wiCoatLocal = coatSf.toLocal(sd.V);
        if (wiBaseLocal.z < kMinCosTheta || wiMidLocal.z < kMinCosTheta || wiCoatLocal.z < kMinCosTheta) return false;

        float4 samplingWeights = layeredTextureSamplingWeights(inputs, baseLayer, midLayer, coatLayer, dustLayer, wiMidLocal, wiCoatLocal);
        float layerSample = sampleNext1D(sg);
        uint layerIndex =
            layerSample < samplingWeights.w ? 3u :
            layerSample < samplingWeights.w + samplingWeights.z ? 2u :
            layerSample < samplingWeights.w + samplingWeights.z + samplingWeights.y ? 1u : 0u;

        float3 woWorld = 0.f.xxx;
        if (layerIndex == 3u)
        {
            float dustPdf;
            float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), dustPdf);
            if (woLocal.z < kMinCosTheta) return false;
            woWorld = coatSf.fromLocal(woLocal);
            result.lobeType = uint(LobeType::DiffuseReflection);
        }
        else if (layerIndex == 2u)
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

        float pdf = layeredTextureEvalCombinedPdf(data, defaultSamplerID, sd, inputs, woWorld);
        if (pdf <= 0.f) return false;

        result.wo = woWorld;
        result.pdf = pdf;
        result.weight = layeredTextureEvalCombined(data, defaultSamplerID, sd, inputs, woWorld) / pdf;
        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        LayeredTextureInputs inputs = sampleLayeredTextureInputs(data, defaultSamplerID, sd);
        return layeredTextureEvalCombinedPdf(data, defaultSamplerID, sd, inputs, wo);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.emission = 0.f.xxx;

        LayeredTextureInputs inputs = sampleLayeredTextureInputs(data, defaultSamplerID, sd);
        LayeredTextureLayerParams baseLayer = layeredTextureGetBaseLayer(data, inputs);
        LayeredTextureLayerParams midLayer = layeredTextureGetMidLayer(data, inputs);
        LayeredTextureLayerParams coatLayer = layeredTextureGetCoatLayer(data, inputs);
        LayeredTextureDustParams dustLayer = layeredTextureGetDustLayer(data, inputs);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, defaultSamplerID, sd, inputs, coatLayer.normalFlatten);

        float totalWeight = max(baseLayer.coverage + midLayer.coverage + coatLayer.coverage + dustLayer.coverage, 1e-6f);
        p.roughness =
            (baseLayer.coverage * baseLayer.roughness + midLayer.coverage * midLayer.roughness + coatLayer.coverage * coatLayer.roughness +
             dustLayer.coverage * dustLayer.roughness) /
            totalWeight;
        p.guideNormal = coatSf.N;
        p.flags = 0;

        p.diffuseReflectionAlbedo =
            (1.f - dustLayer.coverage) * baseLayer.coverage * saturate(inputs.baseColor) +
            dustLayer.coverage * dustLayer.color;
        p.diffuseTransmissionAlbedo = 0.f.xxx;
        p.specularReflectionAlbedo =
            (baseLayer.coverage * baseLayer.F0 + midLayer.coverage * midLayer.F0 + coatLayer.coverage * coatLayer.F0) /
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

struct ThreeLayeredGGXNoHeightMaterial : MaterialBase, IMaterial, IBootstrapFeatureMaterial, ITrainingDirectionGuardMaterial
{
    ThreeLayeredGGXNoHeightMaterialData data;

    typedef ThreeLayeredGGXNoHeightMaterialInstance MaterialInstance;

    ThreeLayeredGGXNoHeightMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        ThreeLayeredGGXNoHeightMaterialInstance m = {};
        m.data = data;
        m.defaultSamplerID = header.getDefaultTextureSamplerID();
        return m;
    }

    [Differentiable]
    ThreeLayeredGGXNoHeightMaterialInstance setupDiffMaterialInstance(
        out DiffMaterialData diffData,
        const MaterialSystem ms,
        const ShadingData sd,
        const ITextureSampler lod,
        const uint hints)
    {
        diffData = {};
        return setupMaterialInstance(ms, sd, lod, hints);
    }

    BootstrapFeatureData getBootstrapFeatures(const MaterialSystem ms, const ShadingData sd)
    {
        BootstrapFeatureData features = {};

        uint samplerID = header.getDefaultTextureSamplerID();
        LayeredTextureInputs inputs = sampleLayeredTextureInputs(data, samplerID, sd);
        LayeredTextureLayerParams baseLayer = layeredTextureGetBaseLayer(data, inputs);
        LayeredTextureLayerParams midLayer = layeredTextureGetMidLayer(data, inputs);
        LayeredTextureLayerParams coatLayer = layeredTextureGetCoatLayer(data, inputs);
        LayeredTextureDustParams dustLayer = layeredTextureGetDustLayer(data, inputs);

        ShadingFrame baseSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, baseLayer.normalFlatten);
        ShadingFrame midSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, midLayer.normalFlatten);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, coatLayer.normalFlatten);

        features.feature0 = float4(saturate(inputs.baseColor), baseLayer.roughness);
        features.feature1 = float4(midLayer.roughness, coatLayer.roughness, baseSf.N.xy);
        features.feature2 = float4(baseSf.N.z, baseSf.T);
        features.feature3 = float4(midSf.N, midSf.T.x);
        features.feature4 = float4(midSf.T.yz, coatSf.N.xy);
        features.feature5 = float4(coatSf.N.z, coatSf.T);
        features.feature6 = float4(dustLayer.coverage, 0.f, 0.f, 0.f);
        return features;
    }

    bool isTrainingDirectionPairValid(const MaterialSystem ms, const ShadingData sd, const float3 wo, const float threshold)
    {
        uint samplerID = header.getDefaultTextureSamplerID();
        LayeredTextureInputs inputs = sampleLayeredTextureInputs(data, samplerID, sd);
        LayeredTextureLayerParams baseLayer = layeredTextureGetBaseLayer(data, inputs);
        LayeredTextureLayerParams midLayer = layeredTextureGetMidLayer(data, inputs);
        LayeredTextureLayerParams coatLayer = layeredTextureGetCoatLayer(data, inputs);

        ShadingFrame baseSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, baseLayer.normalFlatten);
        ShadingFrame midSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, midLayer.normalFlatten);
        ShadingFrame coatSf = layeredTextureShadingFrame(data, samplerID, sd, inputs, coatLayer.normalFlatten);

        float3 wiBaseLocal = baseSf.toLocal(sd.V);
        float3 woBaseLocal = baseSf.toLocal(wo);
        float3 wiMidLocal = midSf.toLocal(sd.V);
        float3 woMidLocal = midSf.toLocal(wo);
        float3 wiCoatLocal = coatSf.toLocal(sd.V);
        float3 woCoatLocal = coatSf.toLocal(wo);

        float minLayerCos = min(
            min(min(wiBaseLocal.z, woBaseLocal.z), min(wiMidLocal.z, woMidLocal.z)),
            min(wiCoatLocal.z, woCoatLocal.z)
        );
        return minLayerCos >= max(threshold, kMinCosTheta);
    }
};
