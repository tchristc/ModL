using ModL.Data.Annotations;

namespace ModL.Data.Datasets;

/// <summary>
/// Scans a ModelNet dataset directory tree and returns catalogued entries.
///
/// Expected layout:
///   {root}/
///     {category}/
///       train/
///         *.off
///       test/
///         *.off
///
/// The catalog respects the dataset's own train/test split when
/// <see cref="Split"/> is set; otherwise it returns all entries.
/// </summary>
public class ModelNetCatalog
{
    public string RootDir { get; }

    private static readonly string[] KnownSplits =
        ["train", "test", "val"];

    public ModelNetCatalog(string rootDir)
    {
        if (!Directory.Exists(rootDir))
            throw new DirectoryNotFoundException($"ModelNet root not found: {rootDir}");
        RootDir = rootDir;
    }

    /// <summary>
    /// Enumerates all model entries, optionally filtered to a specific split.
    /// </summary>
    /// <param name="split">
    /// "train", "test", "val", or null for all entries.
    /// </param>
    public IEnumerable<CatalogEntry> Enumerate(string? split = null)
    {
        // Each immediate sub-directory is a category
        foreach (var categoryDir in Directory.EnumerateDirectories(RootDir).OrderBy(d => d))
        {
            var category = Path.GetFileName(categoryDir);

            // Sub-directories may be split folders or model folders directly
            var subDirs = Directory.EnumerateDirectories(categoryDir).ToArray();
            bool hasSplitFolders = subDirs.Any(d =>
                KnownSplits.Contains(Path.GetFileName(d), StringComparer.OrdinalIgnoreCase));

            if (hasSplitFolders)
            {
                foreach (var splitDir in subDirs.OrderBy(d => d))
                {
                    var splitName = Path.GetFileName(splitDir);
                    if (split != null &&
                        !splitName.Equals(split, StringComparison.OrdinalIgnoreCase))
                        continue;

                    foreach (var entry in EnumerateOffFiles(splitDir, category, splitName))
                        yield return entry;
                }
            }
            else
            {
                // Models sit directly in the category folder
                if (split != null) continue; // no split info, skip when filtered

                foreach (var entry in EnumerateOffFiles(categoryDir, category, "unknown"))
                    yield return entry;
            }
        }
    }

    /// <summary>Returns only the training split entries.</summary>
    public IEnumerable<CatalogEntry> Train => Enumerate("train");

    /// <summary>Returns only the test split entries.</summary>
    public IEnumerable<CatalogEntry> TestSet => Enumerate("test");

    /// <summary>Returns category counts across all entries.</summary>
    public IReadOnlyDictionary<string, int> CategoryCounts(string? split = null)
        => Enumerate(split)
            .GroupBy(e => e.Annotation.Category ?? "unknown")
            .ToDictionary(g => g.Key, g => g.Count());

    // -----------------------------------------------------------------------

    private static IEnumerable<CatalogEntry> EnumerateOffFiles(
        string dir, string category, string split)
    {
        foreach (var file in Directory.EnumerateFiles(dir, "*.off").OrderBy(f => f))
        {
            var ann = new ModelAnnotation
            {
                ModelId  = Path.GetFileNameWithoutExtension(file),
                Category = category,
                Tags     = new[] { category },
                CustomData = new Dictionary<string, object>
                {
                    ["split"]  = split,
                    ["source"] = "ModelNet"
                }
            };
            yield return new CatalogEntry(file, ann);
        }
    }
}

/// <summary>A single model file paired with its derived annotation.</summary>
public record CatalogEntry(string FilePath, ModelAnnotation Annotation);
