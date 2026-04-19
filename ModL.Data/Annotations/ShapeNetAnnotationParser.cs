namespace ModL.Data.Annotations;

/// <summary>
/// Derives annotations from ShapeNet's taxonomy.json and folder structure.
///
/// ShapeNetCore layout:
///   {root}/
///     taxonomy.json               ← maps synsetId → name + children
///     02691156/                   ← synset id (e.g. "airplane")
///       {modelId}/
///         models/
///           model_normalized.obj
/// </summary>
public class ShapeNetAnnotationParser : IAnnotationParser
{
    private readonly Dictionary<string, string> _synsetToName;

    public ShapeNetAnnotationParser(string datasetRootPath)
    {
        _synsetToName = LoadTaxonomy(datasetRootPath);
    }

    public bool CanParse(string filePath)
        => Path.GetExtension(filePath).Equals(".obj", StringComparison.OrdinalIgnoreCase);

    public ModelAnnotation Parse(string modelFilePath)
    {
        // Walk up the path to find the synset directory (8-digit numeric folder)
        var dir = Path.GetDirectoryName(modelFilePath);
        string? synsetId = null;
        string? modelId  = null;

        while (dir != null)
        {
            var name = Path.GetFileName(dir);
            if (name != null && name.Length == 8 && name.All(char.IsDigit))
            {
                synsetId = name;
                // The model id is one level below the synset
                if (modelId == null)
                    modelId = Path.GetFileName(Path.GetDirectoryName(modelFilePath));
                break;
            }
            modelId = name;
            dir = Path.GetDirectoryName(dir);
        }

        var category = synsetId != null && _synsetToName.TryGetValue(synsetId, out var n)
            ? n
            : synsetId ?? "unknown";

        return new ModelAnnotation
        {
            ModelId  = modelId ?? Path.GetFileNameWithoutExtension(modelFilePath),
            Category = category,
            Tags     = new[] { category },
            CustomData = new Dictionary<string, object>
            {
                ["synsetId"] = synsetId ?? "unknown",
                ["source"]   = "ShapeNet"
            }
        };
    }

    private static Dictionary<string, string> LoadTaxonomy(string rootPath)
    {
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var taxonomyPath = Path.Combine(rootPath, "taxonomy.json");

        if (!File.Exists(taxonomyPath))
            return map;

        try
        {
            var json  = File.ReadAllText(taxonomyPath);
            var nodes = Newtonsoft.Json.JsonConvert.DeserializeObject<TaxonomyEntry[]>(json);
            if (nodes == null) return map;

            foreach (var node in nodes)
            {
                if (!string.IsNullOrWhiteSpace(node.SynsetId) && !string.IsNullOrWhiteSpace(node.Name))
                    map[node.SynsetId] = node.Name;
            }
        }
        catch { /* malformed taxonomy – proceed without names */ }

        return map;
    }

    private sealed class TaxonomyEntry
    {
        [Newtonsoft.Json.JsonProperty("synsetId")]
        public string SynsetId { get; set; } = string.Empty;

        [Newtonsoft.Json.JsonProperty("name")]
        public string Name { get; set; } = string.Empty;
    }
}
