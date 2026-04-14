using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace ModL.Data.Annotations;

/// <summary>
/// Parses JSON-based annotations (ShapeNet, PartNet format)
/// </summary>
public class JsonAnnotationParser : IAnnotationParser
{
    public bool CanParse(string filePath)
    {
        return Path.GetExtension(filePath).Equals(".json", StringComparison.OrdinalIgnoreCase);
    }

    public ModelAnnotation Parse(string annotationFile)
    {
        if (!File.Exists(annotationFile))
            throw new FileNotFoundException($"Annotation file not found: {annotationFile}");

        var json = File.ReadAllText(annotationFile);
        var jObject = JObject.Parse(json);

        var annotation = new ModelAnnotation();

        // Parse common fields
        annotation.ModelId = jObject["modelId"]?.ToString() ?? 
                            jObject["id"]?.ToString() ?? 
                            Path.GetFileNameWithoutExtension(annotationFile);

        annotation.Category = jObject["category"]?.ToString() ?? 
                             jObject["label"]?.ToString() ?? 
                             "unknown";

        // Parse tags
        var tagsToken = jObject["tags"] ?? jObject["labels"];
        if (tagsToken != null)
        {
            annotation.Tags = tagsToken.ToObject<string[]>() ?? Array.Empty<string>();
        }

        // Parse part labels
        var partsToken = jObject["parts"] ?? jObject["partLabels"];
        if (partsToken != null)
        {
            annotation.PartLabels = partsToken.ToObject<Dictionary<string, string>>() ?? new();
        }

        // Parse bounding box
        var bboxToken = jObject["boundingBox"] ?? jObject["bbox"];
        if (bboxToken != null)
        {
            annotation.BoundingBox = bboxToken.ToObject<BoundingBoxAnnotation>();
        }

        // Store any custom data
        foreach (var prop in jObject.Properties())
        {
            if (!IsStandardProperty(prop.Name))
            {
                annotation.CustomData[prop.Name] = prop.Value.ToString();
            }
        }

        return annotation;
    }

    private bool IsStandardProperty(string name)
    {
        return name switch
        {
            "modelId" or "id" or "category" or "label" or "tags" or "labels" or 
            "parts" or "partLabels" or "boundingBox" or "bbox" => true,
            _ => false
        };
    }
}

/// <summary>
/// Factory for creating annotation parsers
/// </summary>
public static class AnnotationParserFactory
{
    private static readonly List<IAnnotationParser> _parsers = new()
    {
        new JsonAnnotationParser()
    };

    public static ModelAnnotation Parse(string annotationFile)
    {
        var parser = _parsers.FirstOrDefault(p => p.CanParse(annotationFile));
        
        if (parser == null)
            throw new NotSupportedException($"No parser found for file: {annotationFile}");

        return parser.Parse(annotationFile);
    }

    public static void RegisterParser(IAnnotationParser parser)
    {
        _parsers.Add(parser);
    }
}
