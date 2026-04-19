namespace ModL.Data.Annotations;

/// <summary>
/// Derives annotations from ModelNet's folder structure.
///
/// ModelNet layout:
///   ModelNet40/
///     chair/
///       train/
///         chair_0001.off
///       test/
///         chair_0002.off
///
/// The category is the grandparent folder name.
/// The split (train/test) is the parent folder name.
/// </summary>
public class ModelNetAnnotationParser : IAnnotationParser
{
    public bool CanParse(string filePath)
        => Path.GetExtension(filePath).Equals(".off", StringComparison.OrdinalIgnoreCase);

    public ModelAnnotation Parse(string modelFilePath)
    {
        // grandparent = category (e.g. "chair")
        // parent      = split    (e.g. "train" | "test")
        var parent      = Path.GetFileName(Path.GetDirectoryName(modelFilePath)) ?? "unknown";
        var grandParent = Path.GetFileName(Path.GetDirectoryName(Path.GetDirectoryName(modelFilePath))) ?? "unknown";

        // If grandparent looks like a known split name, swap: category is one level higher
        bool parentIsSplit = parent.Equals("train", StringComparison.OrdinalIgnoreCase)
                          || parent.Equals("test",  StringComparison.OrdinalIgnoreCase)
                          || parent.Equals("val",   StringComparison.OrdinalIgnoreCase);

        string category = parentIsSplit ? grandParent : parent;
        string split    = parentIsSplit ? parent      : "unknown";

        return new ModelAnnotation
        {
            ModelId  = Path.GetFileNameWithoutExtension(modelFilePath),
            Category = category,
            Tags     = new[] { category },
            CustomData = new Dictionary<string, object>
            {
                ["split"]  = split,
                ["source"] = "ModelNet"
            }
        };
    }
}
