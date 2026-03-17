#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include <array>
using namespace Falcor;

class NeuralDecodePass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(NeuralDecodePass, "NeuralDecodePass", "Decode 8D latent texture into RGB");

    static ref<NeuralDecodePass> create(ref<Device> pDevice, const Properties& props);

    NeuralDecodePass(ref<Device> pDevice, const Properties& props);

    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;


    
private:
    enum class DisplayMode : uint32_t
    {
        Raw = 0,
        Clamp01 = 1,
        Log1pPos = 2,
        Abs = 3,
    };

    enum class DebugTensor : uint32_t
    {
        FinalDecoded = 0,
        RawY = 1,
        Z = 2,
        Frame = 3,
        X = 4,
        H0PreRelu = 5,
        H0PostRelu = 6,
        H1PreRelu = 7,
        H1PostRelu = 8,
    };

    void createProgram();
    void loadWeights(const std::string& npzPath);

    ref<Device> mpDevice;
    ref<ComputePass> mpPass;

    ref<Texture> mpLatent0;
    ref<Texture> mpLatent1;

    ref<Buffer> mpFrameLinear;
    ref<Buffer> mpW0;
    ref<Buffer> mpB0;
    ref<Buffer> mpW1;
    ref<Buffer> mpB1;
    ref<Buffer> mpW2;
    ref<Buffer> mpB2;

    float mExpOffset = 3.0f;
    bool mApplyExp = true;
    bool mFlipY = false;
    DisplayMode mDisplayMode = DisplayMode::Raw;

    DebugTensor mDebugTensor = DebugTensor::FinalDecoded;
    uint32_t mDebugBaseChannel = 0;

    uint2 mOutputDim = {512, 512};

    static constexpr uint32_t kDebugSampleCount = 5;

    bool mDumpSamplePixels = false;

    ref<Buffer> mpDebugSampleBuffer;
    ref<Buffer> mpDebugSampleReadbackBuffer;

    std::array<uint2, kDebugSampleCount> mDebugCoords = {
        uint2(0, 0),
        uint2(0, 0),
        uint2(0, 0),
        uint2(0, 0),
        uint2(0, 0),
    };

    void updateDebugCoords();
    void printDebugSamples();
};