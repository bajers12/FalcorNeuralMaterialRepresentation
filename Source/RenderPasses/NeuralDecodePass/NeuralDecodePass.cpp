#include "NeuralDecodePass.h"

namespace
{
    const char kOutput[] = "output";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NeuralDecodePass>();
}

ref<NeuralDecodePass> NeuralDecodePass::create(ref<Device> pDevice, const Properties& props)
{
    return make_ref<NeuralDecodePass>(pDevice, props);
}

NeuralDecodePass::NeuralDecodePass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
    , mpDevice(pDevice)
{
    if (props.has("applyExp"))         mApplyExp = (bool)props["applyExp"];
    if (props.has("expOffset"))        mExpOffset = (float)props["expOffset"];
    if (props.has("flipY"))            mFlipY = (bool)props["flipY"];
    if (props.has("displayMode"))      mDisplayMode = (DisplayMode)(uint32_t)props["displayMode"];
    if (props.has("debugTensor"))      mDebugTensor = (DebugTensor)(uint32_t)props["debugTensor"];
    if (props.has("debugBaseChannel")) mDebugBaseChannel = (uint32_t)props["debugBaseChannel"];
    createProgram();

    const std::string basePath = "C:/Users/s204795/FalcorNeuralMaterialRepresentation/media/NeuralMaterials/CastIronBlue512_2,5Samples/";

    mpLatent0 = Texture::createFromFile(mpDevice, basePath + "latent0.exr", false, false);
    mpLatent1 = Texture::createFromFile(mpDevice, basePath + "latent1.exr", false, false);

    if (!mpLatent0)
        throw std::runtime_error("Failed to load latent texture: " + basePath + "latent0.exr");

    if (!mpLatent1)
        throw std::runtime_error("Failed to load latent texture: " + basePath + "latent1.exr");

    loadWeights(basePath + "decoder_weights.bin");

    mOutputDim = uint2(mpLatent0->getWidth(), mpLatent0->getHeight());
    updateDebugCoords();

    mpDebugSampleBuffer = make_ref<Buffer>(
        mpDevice,
        sizeof(float4),
        kDebugSampleCount,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );

    mpDebugSampleReadbackBuffer = make_ref<Buffer>(
        mpDevice,
        sizeof(float4),
        kDebugSampleCount,
        ResourceBindFlags::None,
        MemoryType::ReadBack,
        nullptr,
        false
    );
}

void NeuralDecodePass::updateDebugCoords()
{
    const uint32_t w = mOutputDim.x;
    const uint32_t h = mOutputDim.y;

    mDebugCoords[0] = uint2(0, 0);
    mDebugCoords[1] = uint2(w / 4, h / 4);
    mDebugCoords[2] = uint2(w / 2, h / 2);
    mDebugCoords[3] = uint2((3 * w) / 4, (3 * h) / 4);
    mDebugCoords[4] = uint2(w - 1, h - 1);
}

void NeuralDecodePass::printDebugSamples()
{
    const float4* pData = reinterpret_cast<const float4*>(mpDebugSampleReadbackBuffer->map());

    logInfo("NeuralDecodePass sample pixels:");

    for (uint32_t i = 0; i < kDebugSampleCount; ++i)
    {
        const auto& c = mDebugCoords[i];
        const auto& v = pData[i];

        logInfo(
            "  ({}, {}) = [{:.9f}, {:.9f}, {:.9f}, {:.9f}]",
            c.x, c.y, v.x, v.y, v.z, v.w
        );
    }

    mpDebugSampleReadbackBuffer->unmap();
}


Properties NeuralDecodePass::getProperties() const
{
    Properties props;
    props["applyExp"] = mApplyExp;
    props["expOffset"] = mExpOffset;
    props["flipY"] = mFlipY;
    props["displayMode"] = (uint32_t)mDisplayMode;
    props["debugTensor"] = (uint32_t)mDebugTensor;
    props["debugBaseChannel"] = mDebugBaseChannel;
    return props;
}

RenderPassReflection NeuralDecodePass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput(kOutput, "Decoded RGB")
        .format(ResourceFormat::RGBA32Float)
        .texture2D(mOutputDim.x, mOutputDim.y);
    return r;
}

void NeuralDecodePass::createProgram()
{
    ProgramDesc desc;
    desc.addShaderLibrary("C:/Users/s204795/FalcorNeuralMaterialRepresentation/Source/RenderPasses/NeuralDecodePass/NeuralDecode.cs.slang").csEntry("main");
    mpPass = ComputePass::create(mpDevice, desc);
}

#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>
static std::vector<float> readFloatArray(std::ifstream& f, size_t count)
{
    std::vector<float> v(count);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
    if (!f) throw std::runtime_error("Failed reading float array from decoder_weights.bin");
    return v;
}

void NeuralDecodePass::loadWeights(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open weight file: " + path);

    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "NMDLWT01", 8) != 0)
        throw std::runtime_error("Invalid weight file magic in: " + path);

    int32_t latentCh = 0;
    int32_t numFrames = 0;
    int32_t applyExp = 0;
    float expOffset = 0.f;

    f.read(reinterpret_cast<char*>(&latentCh), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&numFrames), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&applyExp), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&expOffset), sizeof(float));

    if (!f) throw std::runtime_error("Failed reading weight file header: " + path);

    mApplyExp = (applyExp != 0);
    mExpOffset = expOffset;

    auto frameLinear = readFloatArray(f, 12 * 8);
    auto w0          = readFloatArray(f, 32 * 20);
    auto b0          = readFloatArray(f, 32);
    auto w1          = readFloatArray(f, 32 * 32);
    auto b1          = readFloatArray(f, 32);
    auto w2          = readFloatArray(f, 3 * 32);
    auto b2          = readFloatArray(f, 3);

    auto makeStructured = [&](const std::vector<float>& data) -> ref<Buffer>
    {
        return make_ref<Buffer>(
            mpDevice,
            sizeof(float),                                   // structSize
            static_cast<uint32_t>(data.size()),             // elementCount
            ResourceBindFlags::ShaderResource,              // bindFlags
            MemoryType::DeviceLocal,                        // memoryType
            data.data(),                                    // initData
            false                                           // createCounter
        );
    };

    mpFrameLinear = makeStructured(frameLinear);
    mpW0 = makeStructured(w0);
    mpB0 = makeStructured(b0);
    mpW1 = makeStructured(w1);
    mpB1 = makeStructured(b1);
    mpW2 = makeStructured(w2);
    mpB2 = makeStructured(b2);
}

void NeuralDecodePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pOutput = renderData.getTexture(kOutput);
    if (!pOutput || !mpLatent0 || !mpLatent1) return;

    auto var = mpPass->getRootVar();

    var["gLatent0"] = mpLatent0;
    var["gLatent1"] = mpLatent1;
    var["gOutput"] = pOutput;

    var["gFrameLinear"] = mpFrameLinear;
    var["gW0"] = mpW0;
    var["gB0"] = mpB0;
    var["gW1"] = mpW1;
    var["gB1"] = mpB1;
    var["gW2"] = mpW2;
    var["gB2"] = mpB2;


    var["Params"]["gOutputDim"] = mOutputDim;
    var["Params"]["gExpOffset"] = mExpOffset;
    var["Params"]["gApplyExp"] = mApplyExp ? 1u : 0u;
    var["Params"]["gFlipY"] = mFlipY ? 1u : 0u;
    var["Params"]["gDisplayMode"] = (uint32_t)mDisplayMode;
    var["Params"]["gDebugTensor"] = (uint32_t)mDebugTensor;
    var["Params"]["gDebugBaseChannel"] = mDebugBaseChannel;

    var["gDebugSamples"] = mpDebugSampleBuffer;

    var["Params"]["gDumpSamplePixels"] = mDumpSamplePixels ? 1u : 0u;

    for (uint32_t i = 0; i < kDebugSampleCount; ++i)
    {
        var["Params"]["gDebugCoords"][i] = mDebugCoords[i];
    }

    mpPass->execute(pRenderContext, mOutputDim.x, mOutputDim.y, 1);

    if (mDumpSamplePixels)
    {
        pRenderContext->copyResource(mpDebugSampleReadbackBuffer.get(), mpDebugSampleBuffer.get());

        printDebugSamples();

        mDumpSamplePixels = false;
    }
}

void NeuralDecodePass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Apply exp(y - offset)", mApplyExp);
    widget.var("Exp offset", mExpOffset, 0.f, 12.f);
    widget.checkbox("Flip Y", mFlipY);

    {
        Gui::DropdownList modeList =
        {
            {0, "Raw"},
            {1, "Clamp [0,1]"},
            {2, "log1p(max(x,0))"},
            {3, "abs(x)"}
        };

        uint32_t mode = (uint32_t)mDisplayMode;
        if (widget.dropdown("Display mode", modeList, mode))
            mDisplayMode = (DisplayMode)mode;
    }

    widget.separator();

    {
        Gui::DropdownList tensorList =
        {
            {0, "Final decoded"},
            {1, "Raw y"},
            {2, "Latent z"},
            {3, "Frame"},
            {4, "Concat x"},
            {5, "h0 pre-ReLU"},
            {6, "h0 post-ReLU"},
            {7, "h1 pre-ReLU"},
            {8, "h1 post-ReLU"},
        };

        uint32_t tensorMode = (uint32_t)mDebugTensor;
        if (widget.dropdown("Debug tensor", tensorList, tensorMode))
            mDebugTensor = (DebugTensor)tensorMode;
    }

    widget.var("Base channel", mDebugBaseChannel, 0u, 31u);

    switch (mDebugTensor)
    {
    case DebugTensor::Z:
        widget.text("Valid base channel: 0..5");
        break;
    case DebugTensor::Frame:
        widget.text("Valid base channel: 0..9");
        break;
    case DebugTensor::X:
        widget.text("Valid base channel: 0..17");
        break;
    case DebugTensor::H0PreRelu:
    case DebugTensor::H0PostRelu:
    case DebugTensor::H1PreRelu:
    case DebugTensor::H1PostRelu:
        widget.text("Valid base channel: 0..29");
        break;
    default:
        widget.text("Final decoded / Raw y use RGB directly");
        break;
    }
    if (widget.button("Dump sample pixels"))
    {
        updateDebugCoords();
        mDumpSamplePixels = true;
    }

    widget.text("Sample coords match Python:");
    for (uint32_t i = 0; i < kDebugSampleCount; ++i)
    {
        widget.text(
            std::to_string(mDebugCoords[i].x) + ", " +
            std::to_string(mDebugCoords[i].y)
        );
    }
}