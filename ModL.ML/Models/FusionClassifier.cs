using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ModL.ML.Models;

/// <summary>
/// Fusion network that combines the voxel latent and multi-view latent into
/// a single embedding, then projects to class logits.
///
/// Data flow:
///   voxelLatent  [B, VoxelLatentDim]   → \
///                                          Concat → [B, VoxelLatentDim+ViewLatentDim]
///   viewLatent   [B, ViewLatentDim]    → /
///       → FC(fusedDim, EmbeddingDim) → BN → ReLU → Dropout
///       → FC(EmbeddingDim, numClasses)   ← logits output
///
/// The embedding (before the final classification head) is also exposed so
/// the Generator can use it as a conditioning signal.
/// </summary>
public sealed class FusionClassifier : Module<Tensor, Tensor, Tensor>
{
    public int EmbeddingDim { get; }
    public int NumClasses   { get; }

    private readonly Sequential _fusion;
    private readonly Linear     _classifier;

    public FusionClassifier(
        int voxelLatentDim  = 256,
        int viewLatentDim   = 256,
        int embeddingDim    = 512,
        int numClasses      = 40,
        double dropout      = 0.4)
        : base(nameof(FusionClassifier))
    {
        EmbeddingDim = embeddingDim;
        NumClasses   = numClasses;

        int fusedDim = voxelLatentDim + viewLatentDim;

        _fusion = Sequential(
            Linear(fusedDim, embeddingDim),
            BatchNorm1d(embeddingDim),
            ReLU(),
            Dropout(dropout),
            Linear(embeddingDim, embeddingDim),
            BatchNorm1d(embeddingDim),
            ReLU(),
            Dropout(dropout)
        );

        _classifier = Linear(embeddingDim, numClasses);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass → class logits [B, NumClasses].
    /// </summary>
    /// <param name="voxelLatent">[B, VoxelLatentDim]</param>
    /// <param name="viewLatent">[B, ViewLatentDim]</param>
    public override Tensor forward(Tensor voxelLatent, Tensor viewLatent)
        => _classifier.forward(Embed(voxelLatent, viewLatent));

    /// <summary>
    /// Returns the fused embedding [B, EmbeddingDim] — use this as a
    /// conditioning vector for the Generator or for nearest-neighbour retrieval.
    /// </summary>
    public Tensor Embed(Tensor voxelLatent, Tensor viewLatent)
    {
        using var fused = cat(new[] { voxelLatent, viewLatent }, dim: 1);
        return _fusion.forward(fused);
    }

    /// <summary>Saves model weights to <paramref name="path"/>.</summary>
    public void SaveWeights(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        this.save(path);
    }

    /// <summary>Loads model weights from <paramref name="path"/>.</summary>
    public void LoadWeights(string path) => this.load(path);
}

/// <summary>
/// Convenience wrapper that holds all three sub-networks as a single unit
/// for save/load and device transfer.
/// </summary>
public sealed class ModLModel : IDisposable
{
    public VoxelEncoder    VoxelEnc   { get; }
    public MultiViewEncoder ViewEnc   { get; }
    public FusionClassifier Fusion    { get; }

    private bool _disposed;

    public ModLModel(
        int numClasses    = 40,
        int voxelLatent   = 256,
        int viewLatent    = 256,
        int embeddingDim  = 512)
    {
        VoxelEnc = new VoxelEncoder(voxelLatent);
        ViewEnc  = new MultiViewEncoder(viewLatent);
        Fusion   = new FusionClassifier(voxelLatent, viewLatent, embeddingDim, numClasses);
    }

    /// <summary>Moves all sub-networks to <paramref name="device"/>.</summary>
    public ModLModel ToDevice(Device device)
    {
        VoxelEnc.to(device);
        ViewEnc.to(device);
        Fusion.to(device);
        return this;
    }

    /// <summary>Sets all sub-networks to training mode.</summary>
    public void Train()
    {
        VoxelEnc.train();
        ViewEnc.train();
        Fusion.train();
    }

    /// <summary>Sets all sub-networks to evaluation mode (disables dropout/BN train stats).</summary>
    public void Eval()
    {
        VoxelEnc.eval();
        ViewEnc.eval();
        Fusion.eval();
    }

    /// <summary>
    /// Forward pass: voxels [B,1,R,R,R] + views [B,V,3,H,W] → logits [B, NumClasses]
    /// </summary>
    public Tensor Forward(Tensor voxels, Tensor views)
    {
        using var vLatent = VoxelEnc.forward(voxels);
        using var aLatent = ViewEnc.forward(views);
        return Fusion.forward(vLatent, aLatent);
    }

    /// <summary>Returns the fused embedding without the classification head.</summary>
    public Tensor Embed(Tensor voxels, Tensor views)
    {
        using var vLatent = VoxelEnc.forward(voxels);
        using var aLatent = ViewEnc.forward(views);
        return Fusion.Embed(vLatent, aLatent);
    }

    /// <summary>Saves all weights to a directory (one .bin per sub-network).</summary>
    public void Save(string dir)
    {
        Directory.CreateDirectory(dir);
        VoxelEnc.save(Path.Combine(dir, "voxel_enc.bin"));
        ViewEnc.save(Path.Combine(dir, "view_enc.bin"));
        Fusion.save(Path.Combine(dir, "fusion.bin"));
    }

    /// <summary>Loads all weights from a directory written by <see cref="Save"/>.</summary>
    public void Load(string dir)
    {
        VoxelEnc.load(Path.Combine(dir, "voxel_enc.bin"));
        ViewEnc.load(Path.Combine(dir, "view_enc.bin"));
        Fusion.load(Path.Combine(dir, "fusion.bin"));
    }

    public void Dispose()
    {
        if (_disposed) return;
        VoxelEnc.Dispose();
        ViewEnc.Dispose();
        Fusion.Dispose();
        _disposed = true;
    }
}
