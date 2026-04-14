namespace ModL.Data.Annotations;

/// <summary>
/// Represents annotations for a 3D model
/// </summary>
public class ModelAnnotation
{
    public string ModelId { get; set; } = string.Empty;
    public string Category { get; set; } = string.Empty;
    public string[] Tags { get; set; } = Array.Empty<string>();
    public Dictionary<string, string> PartLabels { get; set; } = new();
    public BoundingBoxAnnotation? BoundingBox { get; set; }
    public Dictionary<string, object> CustomData { get; set; } = new();
}

/// <summary>
/// Bounding box annotation
/// </summary>
public class BoundingBoxAnnotation
{
    public float MinX { get; set; }
    public float MinY { get; set; }
    public float MinZ { get; set; }
    public float MaxX { get; set; }
    public float MaxY { get; set; }
    public float MaxZ { get; set; }
}

/// <summary>
/// Interface for parsing model annotations
/// </summary>
public interface IAnnotationParser
{
    /// <summary>
    /// Parses annotation from a file
    /// </summary>
    ModelAnnotation Parse(string annotationFile);

    /// <summary>
    /// Checks if this parser can handle the given file format
    /// </summary>
    bool CanParse(string filePath);
}
