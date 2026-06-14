import Scene.Material.MaterialSystem;
__exported import Scene.ShadingData;

struct BootstrapFeatureData
{
    float4 feature0;
    float4 feature1;
    float4 feature2;
    float4 feature3;
    float4 feature4;
    float4 feature5;
    float4 feature6;
    float4 feature7;
};

[anyValueSize(128)]
interface IBootstrapFeatureMaterial
{
    BootstrapFeatureData getBootstrapFeatures(const MaterialSystem ms, const ShadingData sd);
}

extension MaterialSystem
{
    IBootstrapFeatureMaterial getBootstrapFeatureMaterial(const uint materialID)
    {
        MaterialType type = getMaterialType(materialID);
        return createDynamicObject<IBootstrapFeatureMaterial, MaterialDataBlob>(int(type), materialData[materialID]);
    }
}
