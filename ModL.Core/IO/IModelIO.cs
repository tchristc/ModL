namespace ModL.Core.IO;

/// <summary>
/// Interface for loading 3D models from files
/// </summary>
public interface IModelLoader
{
    /// <summary>
    /// Loads a 3D model from the specified file path
    /// </summary>
    Geometry.Model3D Load(string filePath);

    /// <summary>
    /// Checks if this loader can load the given file extension
    /// </summary>
    bool CanLoad(string extension);

    /// <summary>
    /// Gets the supported file extensions
    /// </summary>
    string[] SupportedExtensions { get; }
}

/// <summary>
/// Interface for exporting 3D models to files
/// </summary>
public interface IModelExporter
{
    /// <summary>
    /// Exports a 3D model to the specified file path
    /// </summary>
    void Export(Geometry.Model3D model, string filePath);

    /// <summary>
    /// Gets the supported file extensions
    /// </summary>
    string[] SupportedExtensions { get; }
}

/// <summary>
/// Factory for creating model loaders and exporters
/// </summary>
public static class ModelIOFactory
{
    private static readonly List<IModelLoader> _loaders = new();
    private static readonly List<IModelExporter> _exporters = new();

    static ModelIOFactory()
    {
        // Register default loaders and exporters
        RegisterLoader(new ObjLoader());
        RegisterLoader(new OffLoader());
        RegisterExporter(new ObjExporter());
    }

    public static void RegisterLoader(IModelLoader loader)
    {
        _loaders.Add(loader);
    }

    public static string[] SupportedLoadExtensions
        => _loaders.SelectMany(l => l.SupportedExtensions).Distinct().ToArray();

    public static void RegisterExporter(IModelExporter exporter)
    {
        _exporters.Add(exporter);
    }

    public static Geometry.Model3D LoadModel(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        var loader = _loaders.FirstOrDefault(l => l.CanLoad(extension));

        if (loader == null)
            throw new NotSupportedException($"No loader found for extension: {extension}");

        return loader.Load(filePath);
    }

    public static void ExportModel(Geometry.Model3D model, string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        var exporter = _exporters.FirstOrDefault(e => e.SupportedExtensions.Contains(extension));

        if (exporter == null)
            throw new NotSupportedException($"No exporter found for extension: {extension}");

        exporter.Export(model, filePath);
    }
}
