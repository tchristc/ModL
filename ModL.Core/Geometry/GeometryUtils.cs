namespace ModL.Core.Geometry;

/// <summary>
/// Geometric utility functions for mesh processing
/// </summary>
public static class GeometryUtils
{
    /// <summary>
    /// Calculates the surface area of a triangle
    /// </summary>
    public static float TriangleArea(System.Numerics.Vector3 v0, System.Numerics.Vector3 v1, System.Numerics.Vector3 v2)
    {
        var edge1 = v1 - v0;
        var edge2 = v2 - v0;
        return System.Numerics.Vector3.Cross(edge1, edge2).Length() * 0.5f;
    }

    /// <summary>
    /// Calculates the total surface area of a mesh
    /// </summary>
    public static float CalculateSurfaceArea(Mesh mesh)
    {
        float area = 0;
        for (int i = 0; i < mesh.Indices.Length; i += 3)
        {
            var v0 = mesh.Vertices[mesh.Indices[i]];
            var v1 = mesh.Vertices[mesh.Indices[i + 1]];
            var v2 = mesh.Vertices[mesh.Indices[i + 2]];
            area += TriangleArea(v0, v1, v2);
        }
        return area;
    }

    /// <summary>
    /// Merges multiple meshes into a single mesh
    /// </summary>
    public static Mesh MergeMeshes(params Mesh[] meshes)
    {
        if (meshes.Length == 0)
            throw new ArgumentException("At least one mesh required");

        if (meshes.Length == 1)
            return meshes[0];

        int totalVertices = meshes.Sum(m => m.Vertices.Length);
        int totalIndices = meshes.Sum(m => m.Indices.Length);

        var merged = new Mesh
        {
            Vertices = new System.Numerics.Vector3[totalVertices],
            Normals = new System.Numerics.Vector3[totalVertices],
            UVs = new System.Numerics.Vector2[totalVertices],
            Indices = new int[totalIndices]
        };

        int vertexOffset = 0;
        int indexOffset = 0;

        foreach (var mesh in meshes)
        {
            Array.Copy(mesh.Vertices, 0, merged.Vertices, vertexOffset, mesh.Vertices.Length);
            Array.Copy(mesh.Normals, 0, merged.Normals, vertexOffset, mesh.Normals.Length);
            if (mesh.UVs.Length > 0)
                Array.Copy(mesh.UVs, 0, merged.UVs, vertexOffset, mesh.UVs.Length);

            for (int i = 0; i < mesh.Indices.Length; i++)
            {
                merged.Indices[indexOffset + i] = mesh.Indices[i] + vertexOffset;
            }

            vertexOffset += mesh.Vertices.Length;
            indexOffset += mesh.Indices.Length;
        }

        return merged;
    }

    /// <summary>
    /// Simplifies a mesh using decimation (basic implementation)
    /// </summary>
    public static Mesh Decimate(Mesh mesh, float targetReduction)
    {
        if (targetReduction <= 0 || targetReduction >= 1)
            throw new ArgumentException("Target reduction must be between 0 and 1");

        // TODO: Implement proper decimation algorithm (e.g., quadric error metrics)
        // For now, return the original mesh
        return mesh;
    }
}
