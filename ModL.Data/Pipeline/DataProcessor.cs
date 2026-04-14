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
/// Processes raw 3D models into ML-ready format
/// </summary>
public class DataProcessor
{
    private readonly Voxelizer _voxelizer = new();
    private readonly MultiViewRenderer _renderer = new();

    /// <summary>
    /// Processes a single model
    /// </summary>
    public ProcessedModel Process(
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
    /// Processes a batch of models in parallel
    /// </summary>
    public async Task<ProcessedModel[]> ProcessBatch(
        IEnumerable<(Model3D Model, ModelAnnotation? Annotation)> models,
        PreprocessingConfig config,
        IProgress<BatchProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var modelList = models.ToList();
        var results = new ProcessedModel[modelList.Count];
        var completed = 0;

        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount,
            CancellationToken = cancellationToken
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
                    Current = currentCompleted,
                    Total = modelList.Count,
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
    public string CurrentModel { get; set; } = string.Empty;
    public double PercentComplete => Total > 0 ? (double)Current / Total * 100 : 0;
}
