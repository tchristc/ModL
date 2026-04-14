using System.Numerics;
using ModL.Core.Geometry;

namespace ModL.Core.Voxel;

/// <summary>
/// Converts meshes to voxel grids
/// </summary>
public class Voxelizer
{
    /// <summary>
    /// Voxelizes a 3D model into a voxel grid
    /// </summary>
    public VoxelGrid Voxelize(Model3D model, int resolution)
    {
        var grid = new VoxelGrid(resolution);
        var bounds = model.BoundingBox;
        
        if (bounds.Size == Vector3.Zero)
            return grid;

        // Calculate voxel size
        var gridSize = Math.Max(bounds.Size.X, Math.Max(bounds.Size.Y, bounds.Size.Z));
        var voxelSize = gridSize / resolution;

        foreach (var mesh in model.Meshes)
        {
            VoxelizeMesh(mesh, grid, bounds.Min, voxelSize, resolution);
        }

        return grid;
    }

    private void VoxelizeMesh(Mesh mesh, VoxelGrid grid, Vector3 minBounds, float voxelSize, int resolution)
    {
        // Iterate through each triangle
        for (int i = 0; i < mesh.Indices.Length; i += 3)
        {
            var v0 = mesh.Vertices[mesh.Indices[i]];
            var v1 = mesh.Vertices[mesh.Indices[i + 1]];
            var v2 = mesh.Vertices[mesh.Indices[i + 2]];

            // Get bounding box of triangle
            var triMin = Vector3.Min(Vector3.Min(v0, v1), v2);
            var triMax = Vector3.Max(Vector3.Max(v0, v1), v2);

            // Convert to voxel coordinates
            var voxelMin = WorldToVoxel(triMin, minBounds, voxelSize);
            var voxelMax = WorldToVoxel(triMax, minBounds, voxelSize);

            // Clamp to grid bounds
            int xMin = Math.Max(0, voxelMin.X);
            int yMin = Math.Max(0, voxelMin.Y);
            int zMin = Math.Max(0, voxelMin.Z);
            int xMax = Math.Min(resolution - 1, voxelMax.X);
            int yMax = Math.Min(resolution - 1, voxelMax.Y);
            int zMax = Math.Min(resolution - 1, voxelMax.Z);

            // Test each voxel in the bounding box
            for (int z = zMin; z <= zMax; z++)
            {
                for (int y = yMin; y <= yMax; y++)
                {
                    for (int x = xMin; x <= xMax; x++)
                    {
                        var voxelCenter = VoxelToWorld(x, y, z, minBounds, voxelSize);
                        if (IsVoxelIntersectingTriangle(voxelCenter, voxelSize, v0, v1, v2))
                        {
                            grid.SetVoxel(x, y, z, true);
                        }
                    }
                }
            }
        }
    }

    private (int X, int Y, int Z) WorldToVoxel(Vector3 worldPos, Vector3 minBounds, float voxelSize)
    {
        var relative = worldPos - minBounds;
        return (
            (int)(relative.X / voxelSize),
            (int)(relative.Y / voxelSize),
            (int)(relative.Z / voxelSize)
        );
    }

    private Vector3 VoxelToWorld(int x, int y, int z, Vector3 minBounds, float voxelSize)
    {
        return minBounds + new Vector3(
            (x + 0.5f) * voxelSize,
            (y + 0.5f) * voxelSize,
            (z + 0.5f) * voxelSize
        );
    }

    private bool IsVoxelIntersectingTriangle(Vector3 voxelCenter, float voxelSize, Vector3 v0, Vector3 v1, Vector3 v2)
    {
        // Simple sphere-triangle intersection test
        var halfSize = voxelSize * 0.5f;
        var radius = halfSize * 1.732f; // sqrt(3) for diagonal

        // Check if triangle is close enough to voxel center
        var closestPoint = ClosestPointOnTriangle(voxelCenter, v0, v1, v2);
        var distance = Vector3.Distance(voxelCenter, closestPoint);

        return distance <= radius;
    }

    private Vector3 ClosestPointOnTriangle(Vector3 point, Vector3 v0, Vector3 v1, Vector3 v2)
    {
        // Project point onto triangle plane
        var edge0 = v1 - v0;
        var edge1 = v2 - v0;
        var v0ToPoint = point - v0;

        var a = Vector3.Dot(edge0, edge0);
        var b = Vector3.Dot(edge0, edge1);
        var c = Vector3.Dot(edge1, edge1);
        var d = Vector3.Dot(edge0, v0ToPoint);
        var e = Vector3.Dot(edge1, v0ToPoint);

        var det = a * c - b * b;
        var s = b * e - c * d;
        var t = b * d - a * e;

        if (s + t <= det)
        {
            if (s < 0)
            {
                if (t < 0)
                {
                    // Region 4
                    s = Math.Clamp(-d / a, 0, 1);
                    t = 0;
                }
                else
                {
                    // Region 3
                    s = 0;
                    t = Math.Clamp(-e / c, 0, 1);
                }
            }
            else if (t < 0)
            {
                // Region 5
                s = Math.Clamp(-d / a, 0, 1);
                t = 0;
            }
            else
            {
                // Region 0 (inside triangle)
                var invDet = 1 / det;
                s *= invDet;
                t *= invDet;
            }
        }
        else
        {
            if (s < 0)
            {
                // Region 2
                s = 0;
                t = 1;
            }
            else if (t < 0)
            {
                // Region 6
                s = 1;
                t = 0;
            }
            else
            {
                // Region 1
                var numer = c + e - b - d;
                if (numer <= 0)
                {
                    s = 0;
                }
                else
                {
                    var denom = a - 2 * b + c;
                    s = numer >= denom ? 1 : numer / denom;
                }
                t = 1 - s;
            }
        }

        return v0 + s * edge0 + t * edge1;
    }

    /// <summary>
    /// Converts a voxel grid to a mesh using Marching Cubes algorithm
    /// </summary>
    public Mesh MarchingCubes(VoxelGrid voxels)
    {
        // TODO: Implement full Marching Cubes algorithm
        // This is a complex algorithm that requires lookup tables
        // For now, return a placeholder
        
        var mesh = new Mesh
        {
            Name = "VoxelMesh"
        };

        // Simple cube-based reconstruction as placeholder
        var vertices = new List<Vector3>();
        var indices = new List<int>();

        for (int z = 0; z < voxels.Resolution; z++)
        {
            for (int y = 0; y < voxels.Resolution; y++)
            {
                for (int x = 0; x < voxels.Resolution; x++)
                {
                    if (voxels.GetVoxel(x, y, z))
                    {
                        AddCube(vertices, indices, x, y, z, 1.0f / voxels.Resolution);
                    }
                }
            }
        }

        mesh.Vertices = vertices.ToArray();
        mesh.Indices = indices.ToArray();
        mesh.CalculateNormals();

        return mesh;
    }

    private void AddCube(List<Vector3> vertices, List<int> indices, int x, int y, int z, float size)
    {
        var baseIndex = vertices.Count;
        var offset = new Vector3(x * size, y * size, z * size);

        // Add 8 vertices of cube
        vertices.Add(offset + new Vector3(0, 0, 0) * size);
        vertices.Add(offset + new Vector3(1, 0, 0) * size);
        vertices.Add(offset + new Vector3(1, 1, 0) * size);
        vertices.Add(offset + new Vector3(0, 1, 0) * size);
        vertices.Add(offset + new Vector3(0, 0, 1) * size);
        vertices.Add(offset + new Vector3(1, 0, 1) * size);
        vertices.Add(offset + new Vector3(1, 1, 1) * size);
        vertices.Add(offset + new Vector3(0, 1, 1) * size);

        // Add 12 triangles (6 faces * 2 triangles)
        int[] cubeIndices = {
            0,1,2, 0,2,3, // Front
            5,4,7, 5,7,6, // Back
            4,0,3, 4,3,7, // Left
            1,5,6, 1,6,2, // Right
            3,2,6, 3,6,7, // Top
            4,5,1, 4,1,0  // Bottom
        };

        foreach (var idx in cubeIndices)
        {
            indices.Add(baseIndex + idx);
        }
    }
}
