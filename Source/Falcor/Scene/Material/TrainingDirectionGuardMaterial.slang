import Scene.Material.MaterialSystem;
__exported import Scene.ShadingData;

[anyValueSize(128)]
interface ITrainingDirectionGuardMaterial
{
    bool isTrainingDirectionPairValid(const MaterialSystem ms, const ShadingData sd, const float3 wo, const float threshold);
}

extension MaterialSystem
{
    ITrainingDirectionGuardMaterial getTrainingDirectionGuardMaterial(const uint materialID)
    {
        MaterialType type = getMaterialType(materialID);
        return createDynamicObject<ITrainingDirectionGuardMaterial, MaterialDataBlob>(int(type), materialData[materialID]);
    }
}
