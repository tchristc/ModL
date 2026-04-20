using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ModL.ML.Models;

/// <summary>
/// 3-D convolutional encoder that maps a voxel occupancy grid of shape
/// [B, 1, R, R, R] to a dense latent vector of shape [B, LatentDim].
///
/// Architecture (default, R=64):
///   Conv3d 1→32  k3 s1 p1  → BN → ReLU → MaxPool3d k2 s2   → 32×32×32
///   Conv3d 32→64 k3 s1 p1  → BN → ReLU → MaxPool3d k2 s2   → 64×16×16
///   Conv3d 64→128 k3 s1 p1 → BN → ReLU → MaxPool3d k2 s2   → 128×8×8
///   Conv3d 128→256 k3 s1 p1→ BN → ReLU → AdaptiveAvgPool3d(2) → 256×2×2×2
///   Flatten → Linear(256*8, LatentDim) → ReLU → Dropout(0.3)
/// </summary>
public sealed class VoxelEncoder : Module<Tensor, Tensor>
{
    public int LatentDim { get; }

    private readonly Sequential _conv;
    private readonly Sequential _head;

    public VoxelEncoder(int latentDim = 256, double dropout = 0.3)
        : base(nameof(VoxelEncoder))
    {
        LatentDim = latentDim;

        _conv = Sequential(
            // Block 1
            Conv3d(1, 32, 3L, padding: 1L),
            BatchNorm3d(32),
            ReLU(),
            MaxPool3d(2L, stride: 2L),

            // Block 2
            Conv3d(32, 64, 3L, padding: 1L),
            BatchNorm3d(64),
            ReLU(),
            MaxPool3d(2L, stride: 2L),

            // Block 3
            Conv3d(64, 128, 3L, padding: 1L),
            BatchNorm3d(128),
            ReLU(),
            MaxPool3d(2L, stride: 2L),

            // Block 4
            Conv3d(128, 256, 3L, padding: 1L),
            BatchNorm3d(256),
            ReLU(),
            AdaptiveAvgPool3d(new long[] { 2, 2, 2 })
        );

        // 256 × 2 × 2 × 2 = 2048 features after flatten
        _head = Sequential(
            Flatten(),
            Linear(256 * 2 * 2 * 2, latentDim),
            ReLU(),
            Dropout(dropout)
        );

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var features = _conv.forward(x);
        return _head.forward(features);
    }
}
