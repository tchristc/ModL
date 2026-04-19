using ModL.Core.Geometry;
using ModL.Core.Voxel;
using ModL.Core.Rendering;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using ModL.Data.Annotations;

namespace ModL.Data.Pipeline;

/// <summary>
/// Configuration for preprocessing pipeline
/// </summary>
public class PreprocessingConfig
{
    public int VoxelResolution { get; set; } = 64;
    public int MultiViewCount { get; set; } = 12;
    public int TextureSize { get; set; } = 1024;
    public bool Normalize { get; set; } = true;
    public bool Center { get; set; } = true;
    public bool CalculateNormals { get; set; } = true;
    public string OutputFormat { get; set; } = "processed";
    /// <summary>
    /// When set, each processed model is saved to this directory via
    /// <see cref="ProcessedModelStore"/>. Leave null to skip persistence.
    /// </summary>
    public string? OutputDir { get; set; }
}

/// <summary>
/// Outcome of processing a single model.
/// </summary>
public record ProcessingResult(
    string ModelId,
    string FilePath,
    bool Success,
    ProcessedModel? Model,
    Exception? Error)
{
    public static ProcessingResult Ok(string filePath, ProcessedModel model)
        => new(model.ModelId, filePath, true, model, null);

    public static ProcessingResult Fail(string filePath, Exception ex)
        => new(Path.GetFileNameWithoutExtension(filePath), filePath, false, null, ex);
}

/// <summary>
/// Represents a fully processed 3D model ready for ML training
/// </summary>
public class ProcessedModel
{
    public VoxelGrid? Voxels { get; set; }
    public Mesh? NormalizedMesh { get; set; }
    public Image<Rgb24>? TextureMap { get; set; }
    public Image<Rgb24>[]? MultiViews { get; set; }
    public ModelAnnotation? Annotation { get; set; }
    public float[]? FeatureVector { get; set; }
    public string ModelId { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Processes raw 3D models into ML-ready format.
/// Optionally persists each result via <see cref="ProcessedModelStore"/>
/// when <see cref="PreprocessingConfig.OutputDir"/> is set.
/// </summary>
public class DataProcessor
{
    private readonly Voxelizer _voxelizer = new();
    private readonly MultiViewRenderer _renderer = new();
    private readonly ProcessedModelStore _store = new();

    /// <summary>
    /// Processes a single model file path. Tries to infer an annotation from
    /// folder structure (ModelNet, ShapeNet) when none is supplied.
    /// Saves the result if <see cref="PreprocessingConfig.OutputDir"/> is set.
    /// </summary>
    public ProcessingResult ProcessFile(
        string filePath,
        ModelAnnotation? annotation,
        PreprocessingConfig config)
    {
        try
        {
            var model = Core.IO.ModelIOFactory.LoadModel(filePath);

            // Fall back to path-derived annotation when none supplied
            annotation ??= AnnotationParserFactory.InferFromPath(filePath);

            var processed = ProcessModel(model, annotation, config);

            if (config.OutputDir != null)
                _store.Save(processed, config.OutputDir);

            return ProcessingResult.Ok(filePath, processed);
        }
        catch (Exception ex)
        {
            return ProcessingResult.Fail(filePath, ex);
        }
    }

    /// <summary>
    /// Processes an already-loaded model. Annotation inference from path is
    /// not available here — call <see cref="ProcessFile"/> when you have a path.
    /// </summary>
    public ProcessedModel Process(
        Model3D model,
        ModelAnnotation? annotation,
        PreprocessingConfig config)
    {
        var processed = ProcessModel(model, annotation, config);

        if (config.OutputDir != null)
            _store.Save(processed, config.OutputDir);

        return processed;
    }

    private ProcessedModel ProcessModel(
        Model3D model,
        ModelAnnotation? annotation,
        PreprocessingConfig config)
    {
        var processed = new ProcessedModel
        {
            ModelId = model.Name,
            Annotation = annotation
        };

        // Normalize model
        if (config.Normalize)
        {
            model.Normalize();
        }

        // Calculate normals if needed
        if (config.CalculateNormals)
        {
            foreach (var mesh in model.Meshes)
            {
                if (mesh.Normals.Length == 0)
                {
                    mesh.CalculateNormals();
                }
            }
        }

        // Voxelize
        processed.Voxels = _voxelizer.Voxelize(model, config.VoxelResolution);

        // Store normalized mesh
        if (model.Meshes.Length > 0)
        {
            // Merge all meshes into one for simplicity
            processed.NormalizedMesh = model.Meshes.Length == 1 
                ? model.Meshes[0] 
                : Core.Geometry.GeometryUtils.MergeMeshes(model.Meshes);
        }

        // Render multi-view images
        var views = _renderer.GetStandardViews(config.MultiViewCount);
        processed.MultiViews = _renderer.RenderViews(model, views);

        // Extract feature vector (basic geometric features)
        processed.FeatureVector = ExtractFeatures(model, processed);

        // Store metadata
        processed.Metadata["voxelResolution"] = config.VoxelResolution;
        processed.Metadata["viewCount"] = config.MultiViewCount;
        processed.Metadata["vertexCount"] = model.Meshes.Sum(m => m.VertexCount);
        processed.Metadata["triangleCount"] = model.Meshes.Sum(m => m.TriangleCount);

        return processed;
    }

    /// <summary>
    /// Processes a batch of file paths in parallel, returning one
    /// <see cref="ProcessingResult"/> per input (successes and failures).
    /// </summary>
    public async Task<ProcessingResult[]> ProcessBatch(
        IEnumerable<(string FilePath, ModelAnnotation? Annotation)> items,
        PreprocessingConfig config,
        IProgress<BatchProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var list    = items.ToList();
        var results = new ProcessingResult[list.Count];
        var completed = 0;
        var failed    = 0;

        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount,
            CancellationToken      = cancellationToken
        };

        await Task.Run(() =>
        {
            Parallel.For(0, list.Count, options, i =>
            {
                var (filePath, annotation) = list[i];
                results[i] = ProcessFile(filePath, annotation, config);

                if (!results[i].Success)
                    Interlocked.Increment(ref failed);

                var currentCompleted = Interlocked.Increment(ref completed);
                progress?.Report(new BatchProgress
                {
                    Current      = currentCompleted,
                    Total        = list.Count,
                    Failed       = failed,
                    CurrentModel = Path.GetFileName(filePath)
                });
            });
        }, cancellationToken);

        return results;
    }

    /// <summary>
    /// Overload that accepts pre-loaded Model3D instances (no path-based annotation
    /// inference or auto-save; use the file-path overload when available).
    /// </summary>
    public async Task<ProcessedModel[]> ProcessBatch(
        IEnumerable<(Model3D Model, ModelAnnotation? Annotation)> models,
        PreprocessingConfig config,
        IProgress<BatchProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var modelList = models.ToList();
        var results   = new ProcessedModel[modelList.Count];
        var completed = 0;

        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount,
            CancellationToken      = cancellationToken
        };

        await Task.Run(() =>
        {
            Parallel.For(0, modelList.Count, options, i =>
            {
                var (model, annotation) = modelList[i];
                results[i] = Process(model, annotation, config);

                var currentCompleted = Interlocked.Increment(ref completed);
                progress?.Report(new BatchProgress
                {
                    Current      = currentCompleted,
                    Total        = modelList.Count,
                    CurrentModel = model.Name
                });
            });
        }, cancellationToken);

        return results;
    }

    private float[] ExtractFeatures(Model3D model, ProcessedModel processed)
    {
        var features = new List<float>();

        // Bounding box features
        var bbox = model.BoundingBox;
        features.Add(bbox.Size.X);
        features.Add(bbox.Size.Y);
        features.Add(bbox.Size.Z);
        features.Add(bbox.Volume);

        // Mesh statistics
        features.Add(model.Meshes.Sum(m => m.VertexCount));
        features.Add(model.Meshes.Sum(m => m.TriangleCount));
        features.Add(model.Meshes.Length);

        // Voxel occupancy
        if (processed.Voxels != null)
        {
            var occupancy = (float)processed.Voxels.OccupiedVoxelCount() / 
                           (processed.Voxels.Resolution * processed.Voxels.Resolution * processed.Voxels.Resolution);
            features.Add(occupancy);
        }

        // Surface area
        var totalArea = model.Meshes.Sum(m => Core.Geometry.GeometryUtils.CalculateSurfaceArea(m));
        features.Add(totalArea);

        return features.ToArray();
    }
}

/// <summary>
/// Progress information for batch processing
/// </summary>
public class BatchProgress
{
    public int Current { get; set; }
    public int Total { get; set; }
    public int Failed { get; set; }
    public string CurrentModel { get; set; } = string.Empty;
    public double PercentComplete => Total > 0 ? (double)Current / Total * 100 : 0;
}
