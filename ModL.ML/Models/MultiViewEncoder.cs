using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ModL.ML.Models;

/// <summary>
/// Multi-view image encoder.  Accepts a batch of V rendered views per model and
/// produces a single fixed-size latent vector per model.
///
/// Input shape : [B, V, C, H, W]  (B=batch, V=views, C=3 RGB, H/W=128)
/// Output shape: [B, LatentDim]
///
/// Strategy: a shared 2-D ResNet-style backbone encodes each view independently
/// → [B*V, LatentDim/2], then the V view vectors are max-pooled across the view
/// dimension → [B, LatentDim/2], followed by a projection MLP → [B, LatentDim].
///
/// Architecture (per-view backbone):
///   Conv2d 3→64  k7 s2 p3  → BN → ReLU → MaxPool2d k3 s2 p1   (64×32×32)
///   ResBlock 64→128  stride2                                     (128×16×16)
///   ResBlock 128→256 stride2                                     (256×8×8)
///   AdaptiveAvgPool2d(4)                                        (256×4×4)
///   Flatten → Linear(256*16, LatentDim/2)
/// </summary>
public sealed class MultiViewEncoder : Module<Tensor, Tensor>
{
    public int LatentDim { get; }

    private readonly Sequential _backbone;
    private readonly Sequential _projection;

    public MultiViewEncoder(int latentDim = 256, double dropout = 0.3)
        : base(nameof(MultiViewEncoder))
    {
        LatentDim = latentDim;
        int half  = latentDim / 2;

        _backbone = Sequential(
            // Stem
            Conv2d(3, 64, 7L, stride: 2L, padding: 3L, bias: false),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(3L, stride: 2L, padding: 1L),

            // Stage 1 — 64→128, downsample
            ResBlock2d(64, 128, stride: 2),

            // Stage 2 — 128→256, downsample
            ResBlock2d(128, 256, stride: 2),

            // Global pool
            AdaptiveAvgPool2d(new long[] { 4, 4 }),
            Flatten(),
            Linear(256 * 4 * 4, half),
            ReLU(),
            Dropout(dropout)
        );

        _projection = Sequential(
            Linear(half, latentDim),
            ReLU(),
            Dropout(dropout)
        );

        RegisterComponents();
    }

    /// <param name="x">Shape [B, V, 3, H, W]</param>
    public override Tensor forward(Tensor x)
    {
        long b = x.shape[0];
        long v = x.shape[1];

        // Encode each view independently
        using var xBV   = x.reshape(b * v, x.shape[2], x.shape[3], x.shape[4]);
        using var feats = _backbone.forward(xBV);              // [B*V, half]
        using var featsV = feats.reshape(b, v, feats.shape[1]); // [B, V, half]

        // Max-pool across view dimension → [B, half]
        using var pooled = featsV.amax(1);

        return _projection.forward(pooled);  // [B, LatentDim]
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// <summary>
    /// Basic residual block: two Conv2d with BN/ReLU.
    /// When stride>1 or channels differ, a 1×1 projection shortcut is added.
    /// Returns a Module that can be registered in a Sequential.
    /// </summary>
    private static ResBlock2dModule ResBlock2d(int inChannels, int outChannels, int stride = 1)
        => new(inChannels, outChannels, stride);
}

/// <summary>Tiny residual block usable inside a Sequential (public for TorchSharp serialisation).</summary>
public sealed class ResBlock2dModule : Module<Tensor, Tensor>
{
    private readonly Sequential _main;
    private readonly Module<Tensor, Tensor>? _shortcut;

    public ResBlock2dModule(int inC, int outC, int stride)
        : base($"ResBlock2d_{inC}_{outC}")
    {
        _main = Sequential(
            Conv2d(inC, outC, 3L, stride: stride, padding: 1L, bias: false),
            BatchNorm2d(outC),
            ReLU(),
            Conv2d(outC, outC, 3L, stride: 1L, padding: 1L, bias: false),
            BatchNorm2d(outC)
        );

        if (stride != 1 || inC != outC)
        {
            _shortcut = Sequential(
                Conv2d(inC, outC, 1L, stride: stride, bias: false),
                BatchNorm2d(outC)
            );
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var main = _main.forward(x);
        var res  = _shortcut != null ? _shortcut.forward(x) : x;
        using var sum = main + res;
        return functional.relu(sum);
    }
}
