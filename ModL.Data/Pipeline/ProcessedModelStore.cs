using System.Text;
using ModL.Core.Voxel;
using ModL.Core.Geometry;
using ModL.Data.Annotations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Png;
using Newtonsoft.Json;

namespace ModL.Data.Pipeline;

/// <summary>
/// Persists and loads <see cref="ProcessedModel"/> records to/from disk.
///
/// On-disk layout per model (one sub-directory per record):
///   {outputRoot}/{modelId}/
///     meta.json       – annotation, feature vector, metadata dictionary
///     voxels.bin      – int32 resolution, then float32[] occupancy grid (row-major x,y,z)
///     mesh.bin        – vertex positions, normals, UVs, index buffer
///     views/
///       view_00.png … view_NN.png
/// </summary>
public class ProcessedModelStore
{
    private const string MetaFile   = "meta.json";
    private const string VoxelsFile = "voxels.bin";
    private const string MeshFile   = "mesh.bin";
    private const string ViewsDir   = "views";

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    public void Save(ProcessedModel model, string outputRoot)
    {
        var dir = Path.Combine(outputRoot, SanitizeName(model.ModelId));
        Directory.CreateDirectory(dir);

        SaveMeta(model, dir);

        if (model.Voxels != null)
            SaveVoxels(model.Voxels, dir);

        if (model.NormalizedMesh != null)
            SaveMesh(model.NormalizedMesh, dir);

        if (model.MultiViews is { Length: > 0 })
            SaveViews(model.MultiViews, dir);
    }

    // -----------------------------------------------------------------------
    // Read
    // -----------------------------------------------------------------------

    public ProcessedModel Load(string modelDir, bool loadViews = true)
    {
        var model = new ProcessedModel
        {
            ModelId = Path.GetFileName(modelDir)
        };

        var metaPath = Path.Combine(modelDir, MetaFile);
        if (File.Exists(metaPath))
            LoadMeta(model, metaPath);

        var voxelPath = Path.Combine(modelDir, VoxelsFile);
        if (File.Exists(voxelPath))
            model.Voxels = LoadVoxels(voxelPath);

        var meshPath = Path.Combine(modelDir, MeshFile);
        if (File.Exists(meshPath))
            model.NormalizedMesh = LoadMesh(meshPath);

        if (loadViews)
        {
            var viewsPath = Path.Combine(modelDir, ViewsDir);
            if (Directory.Exists(viewsPath))
                model.MultiViews = LoadViews(viewsPath);
        }

        return model;
    }

    /// <summary>
    /// Enumerates every model directory stored under <paramref name="outputRoot"/>.
    /// </summary>
    public IEnumerable<ProcessedModel> LoadAll(string outputRoot, bool loadViews = false)
    {
        foreach (var dir in Directory.EnumerateDirectories(outputRoot))
        {
            ProcessedModel? record = null;
            try { record = Load(dir, loadViews); }
            catch { /* skip corrupt records */ }
            if (record != null) yield return record;
        }
    }

    /// <summary>
    /// Returns the list of all stored model directories without loading them.
    /// </summary>
    public static IReadOnlyList<string> ListModelDirs(string outputRoot)
        => Directory.Exists(outputRoot)
            ? Directory.GetDirectories(outputRoot)
            : Array.Empty<string>();

    // -----------------------------------------------------------------------
    // Meta serialisation
    // -----------------------------------------------------------------------

    private static void SaveMeta(ProcessedModel model, string dir)
    {
        var dto = new MetaDto
        {
            ModelId       = model.ModelId,
            Category      = model.Annotation?.Category,
            Tags          = model.Annotation?.Tags,
            PartLabels    = model.Annotation?.PartLabels,
            CustomData    = model.Annotation?.CustomData,
            FeatureVector = model.FeatureVector,
            Metadata      = model.Metadata
        };

        File.WriteAllText(
            Path.Combine(dir, MetaFile),
            JsonConvert.SerializeObject(dto, Formatting.Indented),
            Encoding.UTF8);
    }

    private static void LoadMeta(ProcessedModel model, string metaPath)
    {
        var json = File.ReadAllText(metaPath, Encoding.UTF8);
        var dto  = JsonConvert.DeserializeObject<MetaDto>(json);
        if (dto == null) return;

        model.ModelId       = dto.ModelId ?? model.ModelId;
        model.FeatureVector = dto.FeatureVector;
        model.Metadata      = dto.Metadata ?? new();

        if (dto.Category != null)
        {
            model.Annotation = new ModelAnnotation
            {
                ModelId    = dto.ModelId ?? string.Empty,
                Category   = dto.Category,
                Tags       = dto.Tags ?? Array.Empty<string>(),
                PartLabels = dto.PartLabels ?? new(),
                CustomData = dto.CustomData ?? new()
            };
        }
    }

    // -----------------------------------------------------------------------
    // Voxel serialisation  (int32 resolution + float32[] flat array)
    // -----------------------------------------------------------------------

    private static void SaveVoxels(VoxelGrid voxels, string dir)
    {
        using var fs = File.Create(Path.Combine(dir, VoxelsFile));
        using var bw = new BinaryWriter(fs);
        bw.Write(voxels.Resolution);
        foreach (var v in voxels.ToFloatArray())
            bw.Write(v);
    }

    private static VoxelGrid LoadVoxels(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        int resolution = br.ReadInt32();
        var grid       = new VoxelGrid(resolution);
        int total      = resolution * resolution * resolution;
        var array      = new float[total];
        for (int i = 0; i < total; i++)
            array[i] = br.ReadSingle();
        grid.FromFloatArray(array);
        return grid;
    }

    // -----------------------------------------------------------------------
    // Mesh serialisation
    // Header: int32 nVerts, int32 nNormals, int32 nUVs, int32 nIndices
    // Then each array as raw float32 / int32 values
    // -----------------------------------------------------------------------

    private static void SaveMesh(Mesh mesh, string dir)
    {
        using var fs = File.Create(Path.Combine(dir, MeshFile));
        using var bw = new BinaryWriter(fs);

        bw.Write(mesh.Vertices.Length);
        bw.Write(mesh.Normals.Length);
        bw.Write(mesh.UVs.Length);
        bw.Write(mesh.Indices.Length);

        foreach (var v in mesh.Vertices)  { bw.Write(v.X); bw.Write(v.Y); bw.Write(v.Z); }
        foreach (var n in mesh.Normals)   { bw.Write(n.X); bw.Write(n.Y); bw.Write(n.Z); }
        foreach (var uv in mesh.UVs)      { bw.Write(uv.X); bw.Write(uv.Y); }
        foreach (var idx in mesh.Indices) bw.Write(idx);
    }

    private static Mesh LoadMesh(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        int nVerts   = br.ReadInt32();
        int nNormals = br.ReadInt32();
        int nUVs     = br.ReadInt32();
        int nIndices = br.ReadInt32();

        var verts   = new System.Numerics.Vector3[nVerts];
        var normals = new System.Numerics.Vector3[nNormals];
        var uvs     = new System.Numerics.Vector2[nUVs];
        var indices = new int[nIndices];

        for (int i = 0; i < nVerts;   i++) verts[i]   = new(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        for (int i = 0; i < nNormals; i++) normals[i]  = new(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        for (int i = 0; i < nUVs;     i++) uvs[i]      = new(br.ReadSingle(), br.ReadSingle());
        for (int i = 0; i < nIndices; i++) indices[i]  = br.ReadInt32();

        return new Mesh
        {
            Vertices = verts,
            Normals  = normals,
            UVs      = uvs,
            Indices  = indices
        };
    }

    // -----------------------------------------------------------------------
    // View image serialisation
    // -----------------------------------------------------------------------

    private static void SaveViews(Image<Rgb24>[] views, string dir)
    {
        var viewsDir = Path.Combine(dir, ViewsDir);
        Directory.CreateDirectory(viewsDir);
        for (int i = 0; i < views.Length; i++)
        {
            using var fs = File.Create(Path.Combine(viewsDir, $"view_{i:D2}.png"));
            views[i].Save(fs, new PngEncoder());
        }
    }

    private static Image<Rgb24>[] LoadViews(string viewsDir)
    {
        var files = Directory.GetFiles(viewsDir, "view_*.png")
            .OrderBy(f => f)
            .ToArray();

        var images = new Image<Rgb24>[files.Length];
        for (int i = 0; i < files.Length; i++)
            images[i] = Image.Load<Rgb24>(files[i]);
        return images;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static string SanitizeName(string name)
        => string.Concat(name.Select(c => Path.GetInvalidFileNameChars().Contains(c) ? '_' : c));

    // -----------------------------------------------------------------------
    // DTO
    // -----------------------------------------------------------------------

    private sealed class MetaDto
    {
        public string?                     ModelId       { get; set; }
        public string?                     Category      { get; set; }
        public string[]?                   Tags          { get; set; }
        public Dictionary<string, string>? PartLabels    { get; set; }
        public Dictionary<string, object>? CustomData    { get; set; }
        public float[]?                    FeatureVector { get; set; }
        public Dictionary<string, object>? Metadata      { get; set; }
    }
}
