namespace ModL.ML.Training;

/// <summary>
/// All hyperparameters and paths needed to train or resume a ModL model.
/// Serialisable to/from JSON for reproducibility.
/// </summary>
public sealed class TrainingConfig
{
    // ── Dataset ───────────────────────────────────────────────────────────
    /// <summary>Root directory written by the preprocess command.</summary>
    public string ProcessedDir { get; set; } = string.Empty;

    /// <summary>Index files written by the split command.  Null = use all data for training.</summary>
    public string? TrainIndexFile { get; set; }
    public string? ValIndexFile   { get; set; }
    public string? TestIndexFile  { get; set; }

    // ── Model ─────────────────────────────────────────────────────────────
    /// <summary>Number of target categories (labels).</summary>
    public int NumClasses      { get; set; } = 40;

    /// <summary>Latent dimension for the voxel encoder output.</summary>
    public int VoxelLatentDim  { get; set; } = 256;

    /// <summary>Latent dimension for the multi-view encoder output.</summary>
    public int ViewLatentDim   { get; set; } = 256;

    /// <summary>Fused embedding dimension (input to the classifier head).</summary>
    public int EmbeddingDim    { get; set; } = 512;

    /// <summary>Dropout probability applied in encoder heads and fusion MLP.</summary>
    public double Dropout      { get; set; } = 0.4;

    // ── Input ─────────────────────────────────────────────────────────────
    /// <summary>Voxel grid side length expected by VoxelEncoder (must match preprocessing).</summary>
    public int VoxelResolution  { get; set; } = 64;

    /// <summary>Number of multi-view images per model (must match preprocessing).</summary>
    public int NumViews          { get; set; } = 12;

    /// <summary>Height/width that view images are resized to before batching.</summary>
    public int ViewImageSize     { get; set; } = 128;

    // ── Training ──────────────────────────────────────────────────────────
    public int    Epochs         { get; set; } = 100;
    public int    BatchSize      { get; set; } = 32;
    public float  LearningRate   { get; set; } = 1e-3f;
    public float  WeightDecay    { get; set; } = 1e-4f;

    /// <summary>
    /// LR schedule: "cosine" (CosineAnnealingLR) or "step" (StepLR).
    /// </summary>
    public string LrSchedule     { get; set; } = "cosine";

    /// <summary>StepLR: decay factor applied every <see cref="LrStepEvery"/> epochs.</summary>
    public float  LrStepGamma    { get; set; } = 0.5f;
    public int    LrStepEvery    { get; set; } = 20;

    /// <summary>Whether to apply random augmentations during training.</summary>
    public bool   Augment        { get; set; } = true;

    // ── Hardware ──────────────────────────────────────────────────────────
    /// <summary>"cpu", "cuda", or "cuda:N" for a specific GPU.</summary>
    public string Device         { get; set; } = "cpu";

    // ── Persistence ───────────────────────────────────────────────────────
    /// <summary>Directory where checkpoints and the best model weights are saved.</summary>
    public string CheckpointDir  { get; set; } = "checkpoints";

    /// <summary>
    /// If set, training resumes from this checkpoint directory instead of
    /// starting fresh.
    /// </summary>
    public string? ResumeFrom    { get; set; }

    /// <summary>Save a checkpoint every N epochs (0 = only save best).</summary>
    public int    SaveEveryEpochs { get; set; } = 10;

    // ── Utilities ─────────────────────────────────────────────────────────
    public static TrainingConfig FromJson(string path)
        => Newtonsoft.Json.JsonConvert.DeserializeObject<TrainingConfig>(
               File.ReadAllText(path))
           ?? throw new InvalidDataException("Cannot parse training config: " + path);

    public void SaveJson(string path)
        => File.WriteAllText(path,
               Newtonsoft.Json.JsonConvert.SerializeObject(this, Newtonsoft.Json.Formatting.Indented));

    /// <summary>Returns a minimal valid config useful for quick smoke-tests.</summary>
    public static TrainingConfig Default(string processedDir) => new()
    {
        ProcessedDir = processedDir,
        Epochs       = 10,
        BatchSize    = 8,
        Device       = "cpu"
    };
}

/// <summary>Metrics recorded at the end of each epoch.</summary>
public sealed record EpochMetrics(
    int    Epoch,
    double TrainLoss,
    double ValLoss,
    double ValTop1Accuracy,
    double ValTop5Accuracy,
    TimeSpan Duration);
