using System.Globalization;
using System.Numerics;
using ModL.Core.Geometry;

namespace ModL.Core.IO;

/// <summary>
/// Loads Object File Format (.off) files used by datasets like ModelNet and Princeton Shape Benchmark.
///
/// OFF format:
///   Line 1:  "OFF"  (optionally with counts on same line: "OFF nVerts nFaces nEdges")
///   Line 2:  nVerts nFaces nEdges
///   Lines:   x y z        (one vertex per line)
///   Lines:   n i0 i1 ...  (n = number of indices in face, followed by 0-based vertex indices)
/// </summary>
public class OffLoader : IModelLoader
{
    public string[] SupportedExtensions => new[] { ".off" };

    public bool CanLoad(string extension)
        => extension.Equals(".off", StringComparison.OrdinalIgnoreCase);

    public Model3D Load(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"OFF file not found: {filePath}");

        using var reader = new StreamReader(filePath);

        // --- Header ---
        var header = ReadNextNonEmptyLine(reader)
            ?? throw new FormatException("Unexpected end of file reading OFF header.");

        int nVerts, nFaces;

        if (header.StartsWith("OFF", StringComparison.OrdinalIgnoreCase))
        {
            // Counts may follow on the same header line: "OFF 8 6 0"
            var rest = header["OFF".Length..].Trim();
            if (rest.Length > 0)
            {
                ParseCounts(rest, out nVerts, out nFaces);
            }
            else
            {
                var countsLine = ReadNextNonEmptyLine(reader)
                    ?? throw new FormatException("Missing vertex/face counts line.");
                ParseCounts(countsLine, out nVerts, out nFaces);
            }
        }
        else
        {
            // Some files omit the "OFF" keyword and start directly with counts
            ParseCounts(header, out nVerts, out nFaces);
        }

        // --- Vertices ---
        var vertices = new Vector3[nVerts];
        for (int i = 0; i < nVerts; i++)
        {
            var line = ReadNextNonEmptyLine(reader)
                ?? throw new FormatException($"Expected vertex {i} but reached end of file.");

            var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 3)
                throw new FormatException($"Vertex line {i} has fewer than 3 components: '{line}'");

            vertices[i] = new Vector3(
                ParseFloat(parts[0]),
                ParseFloat(parts[1]),
                ParseFloat(parts[2]));
        }

        // --- Faces (triangulated) ---
        var indices = new List<int>(nFaces * 3);
        for (int i = 0; i < nFaces; i++)
        {
            var line = ReadNextNonEmptyLine(reader)
                ?? throw new FormatException($"Expected face {i} but reached end of file.");

            var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 4)
                throw new FormatException($"Face line {i} is too short: '{line}'");

            int n = int.Parse(parts[0], CultureInfo.InvariantCulture);
            if (parts.Length < n + 1)
                throw new FormatException($"Face line {i} declares {n} vertices but only has {parts.Length - 1}.");

            // Fan triangulation for polygons with more than 3 vertices
            int v0 = int.Parse(parts[1], CultureInfo.InvariantCulture);
            for (int t = 1; t < n - 1; t++)
            {
                int v1 = int.Parse(parts[1 + t], CultureInfo.InvariantCulture);
                int v2 = int.Parse(parts[2 + t], CultureInfo.InvariantCulture);
                indices.Add(v0);
                indices.Add(v1);
                indices.Add(v2);
            }
        }

        var mesh = new Mesh
        {
            Name = Path.GetFileNameWithoutExtension(filePath),
            Vertices = vertices,
            Indices = indices.ToArray(),
            Normals = Array.Empty<Vector3>(),
            UVs = Array.Empty<System.Numerics.Vector2>()
        };

        mesh.CalculateNormals();

        var model = new Model3D
        {
            Name = Path.GetFileNameWithoutExtension(filePath),
            Meshes = new[] { mesh },
            Materials = Array.Empty<Material>()
        };

        model.CalculateBoundingBox();
        return model;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static void ParseCounts(string line, out int nVerts, out int nFaces)
    {
        var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 2)
            throw new FormatException($"Expected 'nVerts nFaces [nEdges]' but got: '{line}'");

        nVerts = int.Parse(parts[0], CultureInfo.InvariantCulture);
        nFaces = int.Parse(parts[1], CultureInfo.InvariantCulture);
    }

    private static string? ReadNextNonEmptyLine(StreamReader reader)
    {
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            line = line.Trim();
            // Skip blank lines and comments (some OFF files use '#')
            if (line.Length > 0 && !line.StartsWith('#'))
                return line;
        }
        return null;
    }

    private static float ParseFloat(string value)
        => float.Parse(value, CultureInfo.InvariantCulture);
}
