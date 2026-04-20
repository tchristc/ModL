using ModL.Data.Pipeline;
using ModL.Data.Annotations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using static TorchSharp.torch;

namespace ModL.ML.Data;

/// <summary>
/// A single row consumed by ML.NET pipelines. All fields are float arrays
/// (features) or a scalar label.
/// </summary>
public sealed class ModelRow
{
    public float[] Features { get; set; } = Array.Empty<float>();
    public float[] Voxels   { get; set; } = Array.Empty<float>();
    public float   Label    { get; set; }
    public string  Category { get; set; } = string.Empty;
    public string  ModelId  { get; set; } = string.Empty;
}

/// <summary>
/// Streams <see cref="ProcessedModel"/> records from disk and batches them
/// into TorchSharp tensors ready for the training loop.
///
/// Each batch contains:
///   voxels  – [B, 1, R, R, R]  float32
///   views   – [B, V, 3, H, W]  float32  (pixel values normalised to [0,1])
///   labels  – [B]               int64
/// </summary>
public sealed class ModelDataLoader : IDisposable
{
    private readonly ProcessedModelStore _store = new();
    private readonly TrainingBatchConfig _cfg;

    public IReadOnlyDictionary<string, int> LabelMap { get; }

    public ModelDataLoader(TrainingBatchConfig cfg, IReadOnlyDictionary<string, int>? labelMap = null)
    {
        _cfg     = cfg;
        LabelMap = labelMap ?? BuildLabelMap(cfg.ProcessedDir, cfg.IndexFile);
    }

    // -----------------------------------------------------------------------
    // Batch enumeration
    // -----------------------------------------------------------------------

    /// <summary>
    /// Iterates over all models in the index (or the whole store if no index)
    /// and yields TorchSharp tensor batches.
    /// </summary>
    public IEnumerable<ModelBatch> GetBatches(bool shuffle = true, int? seed = null)
    {
        var dirs    = GetModelDirs();
        if (shuffle) dirs = Shuffle(dirs, seed ?? Environment.TickCount);

        var buffer  = new List<ProcessedModel>(_cfg.BatchSize);

        foreach (var dir in dirs)
        {
            ProcessedModel record;
            try { record = _store.Load(dir, loadViews: true); }
            catch { continue; }

            buffer.Add(record);

            if (buffer.Count >= _cfg.BatchSize)
            {
                var batch = BuildBatch(buffer);
                buffer.Clear();
                yield return batch;
            }
        }

        if (buffer.Count > 0)
            yield return BuildBatch(buffer);
    }

    // -----------------------------------------------------------------------
    // Tensor construction
    // -----------------------------------------------------------------------

    private ModelBatch BuildBatch(IList<ProcessedModel> records)
    {
        int b   = records.Count;
        int r   = _cfg.VoxelResolution;
        int v   = _cfg.NumViews;
        int h   = _cfg.ViewImageSize;
        int w   = _cfg.ViewImageSize;

        var voxelData  = new float[b * r * r * r];
        var viewData   = new float[b * v * 3 * h * w];
        var labelData  = new long[b];

        for (int i = 0; i < b; i++)
        {
            var rec = records[i];

            // Voxels ─────────────────────────────────────────────────────
            if (rec.Voxels != null)
            {
                var flat = rec.Voxels.ToFloatArray();
                int expected = r * r * r;
                if (flat.Length == expected)
                    flat.CopyTo(voxelData, i * expected);
            }

            // Views ──────────────────────────────────────────────────────
            var views = rec.MultiViews;
            if (views != null)
            {
                int numActual = Math.Min(views.Length, v);
                for (int vi = 0; vi < numActual; vi++)
                {
                    var img = ResizeView(views[vi], h, w);
                    WriteViewPixels(img, viewData, i, vi, v, h, w);
                }
            }

            // Label ──────────────────────────────────────────────────────
            var cat = rec.Annotation?.Category ?? "unknown";
            labelData[i] = LabelMap.TryGetValue(cat, out int lbl) ? lbl : 0;
        }

        return new ModelBatch(
            tensor(voxelData).reshape(b, 1, r, r, r),
            tensor(viewData).reshape(b, v, 3, h, w),
            tensor(labelData));
    }

    private static Image<Rgb24> ResizeView(Image<Rgb24> src, int h, int w)
    {
        if (src.Width == w && src.Height == h) return src;
        return src.Clone(ctx => ctx.Resize(new SixLabors.ImageSharp.Processing.ResizeOptions
            { Size = new SixLabors.ImageSharp.Size(w, h) }));
    }

    private static void WriteViewPixels(
        Image<Rgb24> img, float[] dst,
        int batchIdx, int viewIdx, int totalViews, int h, int w)
    {
        int bvBase = (batchIdx * totalViews + viewIdx) * 3 * h * w;
        for (int y = 0; y < h; y++)
        {
            var row = img.Frames.RootFrame.PixelBuffer.DangerousGetRowSpan(y);
            for (int x = 0; x < w; x++)
            {
                var px = row[x];
                int pBase = bvBase + y * w + x;
                dst[pBase]                 = px.R / 255f;  // R channel
                dst[pBase + h * w]         = px.G / 255f;  // G channel
                dst[pBase + 2 * h * w]     = px.B / 255f;  // B channel
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private List<string> GetModelDirs()
    {
        var all = ProcessedModelStore.ListModelDirs(_cfg.ProcessedDir);
        if (_cfg.IndexFile == null) return all.ToList();

        if (!File.Exists(_cfg.IndexFile)) return all.ToList();
        var allowed = File.ReadAllLines(_cfg.IndexFile)
            .Select(l => l.Trim())
            .Where(l => l.Length > 0)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        return all.Where(d => allowed.Contains(Path.GetFileName(d))).ToList();
    }

    private static List<string> Shuffle(List<string> list, int seed)
    {
        var rng = new Random(seed);
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
        return list;
    }

    private IReadOnlyDictionary<string, int> BuildLabelMap(string processedDir, string? indexFile)
    {
        var dirs    = GetModelDirs();
        var cats    = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var dir in dirs)
        {
            try
            {
                var m = _store.Load(dir, loadViews: false);
                cats.Add(m.Annotation?.Category ?? "unknown");
            }
            catch { }
        }

        return cats.Select((c, i) => (c, i))
                   .ToDictionary(t => t.c, t => t.i);
    }

    public void Dispose() { }
}

/// <summary>Configuration for batching during training/evaluation.</summary>
public sealed class TrainingBatchConfig
{
    public string  ProcessedDir   { get; set; } = string.Empty;
    public string? IndexFile      { get; set; }
    public int     BatchSize      { get; set; } = 32;
    public int     VoxelResolution{ get; set; } = 64;
    public int     NumViews       { get; set; } = 12;
    public int     ViewImageSize  { get; set; } = 128;
}

/// <summary>One mini-batch of tensors.</summary>
public sealed record ModelBatch(Tensor Voxels, Tensor Views, Tensor Labels) : IDisposable
{
    public void Dispose()
    {
        Voxels.Dispose();
        Views.Dispose();
        Labels.Dispose();
    }
}
