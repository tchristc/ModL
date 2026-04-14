using System.Numerics;

namespace ModL.Core.Geometry;

/// <summary>
/// Represents a complete 3D model with meshes, materials, and metadata
/// </summary>
public class Model3D
{
    public Mesh[] Meshes { get; set; } = Array.Empty<Mesh>();
    public Material[] Materials { get; set; } = Array.Empty<Material>();
    public Dictionary<string, object> Metadata { get; set; } = new();
    public string Name { get; set; } = string.Empty;
    public BoundingBox BoundingBox { get; private set; }

    public void CalculateBoundingBox()
    {
        if (Meshes.Length == 0)
        {
            BoundingBox = new BoundingBox(Vector3.Zero, Vector3.Zero);
            return;
        }

        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);

        foreach (var mesh in Meshes)
        {
            foreach (var vertex in mesh.Vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }
        }

        BoundingBox = new BoundingBox(min, max);
    }

    public void Normalize()
    {
        CalculateBoundingBox();
        var center = BoundingBox.Center;
        var size = BoundingBox.Size;
        var scale = 1.0f / Math.Max(size.X, Math.Max(size.Y, size.Z));

        foreach (var mesh in Meshes)
        {
            for (int i = 0; i < mesh.Vertices.Length; i++)
            {
                mesh.Vertices[i] = (mesh.Vertices[i] - center) * scale;
            }
        }

        CalculateBoundingBox();
    }
}

/// <summary>
/// Represents a single mesh with vertices, indices, normals, and UV coordinates
/// </summary>
public class Mesh
{
    public Vector3[] Vertices { get; set; } = Array.Empty<Vector3>();
    public int[] Indices { get; set; } = Array.Empty<int>();
    public Vector3[] Normals { get; set; } = Array.Empty<Vector3>();
    public Vector2[] UVs { get; set; } = Array.Empty<Vector2>();
    public int MaterialIndex { get; set; } = -1;
    public string Name { get; set; } = string.Empty;

    public int VertexCount => Vertices.Length;
    public int TriangleCount => Indices.Length / 3;

    public void CalculateNormals()
    {
        if (Vertices.Length == 0 || Indices.Length == 0)
            return;

        Normals = new Vector3[Vertices.Length];

        for (int i = 0; i < Indices.Length; i += 3)
        {
            int i0 = Indices[i];
            int i1 = Indices[i + 1];
            int i2 = Indices[i + 2];

            var v0 = Vertices[i0];
            var v1 = Vertices[i1];
            var v2 = Vertices[i2];

            var normal = Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));

            Normals[i0] += normal;
            Normals[i1] += normal;
            Normals[i2] += normal;
        }

        for (int i = 0; i < Normals.Length; i++)
        {
            Normals[i] = Vector3.Normalize(Normals[i]);
        }
    }
}

/// <summary>
/// Represents a material with color and texture properties
/// </summary>
public class Material
{
    public string Name { get; set; } = string.Empty;
    public Vector4 DiffuseColor { get; set; } = new Vector4(1, 1, 1, 1);
    public string? DiffuseTexture { get; set; }
    public float Roughness { get; set; } = 0.5f;
    public float Metallic { get; set; } = 0.0f;
    public Vector3 EmissiveColor { get; set; } = Vector3.Zero;
    public string? NormalMap { get; set; }
    public string? RoughnessMap { get; set; }
    public string? MetallicMap { get; set; }
}

/// <summary>
/// Represents an axis-aligned bounding box
/// </summary>
public readonly struct BoundingBox
{
    public Vector3 Min { get; }
    public Vector3 Max { get; }
    
    public BoundingBox(Vector3 min, Vector3 max)
    {
        Min = min;
        Max = max;
    }

    public Vector3 Center => (Min + Max) * 0.5f;
    public Vector3 Size => Max - Min;
    public float Volume => Size.X * Size.Y * Size.Z;

    public bool Contains(Vector3 point)
    {
        return point.X >= Min.X && point.X <= Max.X &&
               point.Y >= Min.Y && point.Y <= Max.Y &&
               point.Z >= Min.Z && point.Z <= Max.Z;
    }

    public bool Intersects(BoundingBox other)
    {
        return (Min.X <= other.Max.X && Max.X >= other.Min.X) &&
               (Min.Y <= other.Max.Y && Max.Y >= other.Min.Y) &&
               (Min.Z <= other.Max.Z && Max.Z >= other.Min.Z);
    }
}
