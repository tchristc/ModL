using System.Numerics;
using ModL.Core.Geometry;
using ModL.Core.Voxel;

namespace ModL.Data.Pipeline;

/// <summary>
/// Data augmentation helpers for 3D models.
/// All methods return new objects — inputs are never mutated.
/// </summary>
public static class DataAugmentation
{
    // -----------------------------------------------------------------------
    // Mesh augmentations
    // -----------------------------------------------------------------------

    /// <summary>
    /// Rotates all mesh vertices and normals around the Y axis by <paramref name="degrees"/>.
    /// </summary>
    public static Mesh RotateY(Mesh mesh, float degrees)
    {
        var m = Matrix4x4.CreateRotationY(degrees * MathF.PI / 180f);
        return TransformMesh(mesh, m);
    }

    /// <summary>
    /// Uniformly scales all mesh vertices by <paramref name="scale"/>.
    /// </summary>
    public static Mesh Scale(Mesh mesh, float scale)
    {
        var m = Matrix4x4.CreateScale(scale);
        return TransformMesh(mesh, m);
    }

    /// <summary>
    /// Mirrors vertices across the X axis (left-right flip).
    /// Flipping reverses winding so indices are reversed per triangle.
    /// </summary>
    public static Mesh FlipX(Mesh mesh)
    {
        var verts = mesh.Vertices.Select(v => new Vector3(-v.X, v.Y, v.Z)).ToArray();
        var norms = mesh.Normals.Select(n => new Vector3(-n.X, n.Y, n.Z)).ToArray();

        // Reverse winding per triangle to maintain outward-facing normals
        var indices = (int[])mesh.Indices.Clone();
        for (int i = 0; i < indices.Length; i += 3)
            (indices[i + 1], indices[i + 2]) = (indices[i + 2], indices[i + 1]);

        return new Mesh
        {
            Vertices = verts,
            Normals  = norms,
            UVs      = mesh.UVs,
            Indices  = indices
        };
    }

    /// <summary>
    /// Randomly rotates the mesh around Y axis by a multiple of <paramref name="stepDegrees"/>.
    /// </summary>
    public static Mesh RandomRotateY(Mesh mesh, Random rng, float stepDegrees = 90f)
    {
        int steps   = (int)(360f / stepDegrees);
        float angle = rng.Next(steps) * stepDegrees;
        return RotateY(mesh, angle);
    }

    /// <summary>
    /// Applies a random scale jitter in [1 - <paramref name="maxJitter"/>, 1 + <paramref name="maxJitter"/>].
    /// </summary>
    public static Mesh RandomScale(Mesh mesh, Random rng, float maxJitter = 0.1f)
    {
        float scale = 1f + (float)(rng.NextDouble() * 2 - 1) * maxJitter;
        return Scale(mesh, scale);
    }

    // -----------------------------------------------------------------------
    // VoxelGrid augmentations
    // -----------------------------------------------------------------------

    /// <summary>
    /// Rotates a voxel grid 90° around the Y axis (counter-clockwise when viewed from above).
    /// 90° increments are lossless for cubic grids.
    /// </summary>
    public static VoxelGrid RotateY90(VoxelGrid grid)
    {
        int r      = grid.Resolution;
        var result = new VoxelGrid(r);

        for (int x = 0; x < r; x++)
        for (int y = 0; y < r; y++)
        for (int z = 0; z < r; z++)
        {
            if (grid.GetVoxel(x, y, z))
                result.SetVoxel(r - 1 - z, y, x, true);
        }

        return result;
    }

    /// <summary>
    /// Rotates by k × 90° around Y (k ∈ {0,1,2,3}).
    /// </summary>
    public static VoxelGrid RotateY90k(VoxelGrid grid, int k)
    {
        k %= 4;
        var current = grid;
        for (int i = 0; i < k; i++)
            current = RotateY90(current);
        return current;
    }

    /// <summary>
    /// Mirrors a voxel grid across the X axis.
    /// </summary>
    public static VoxelGrid FlipX(VoxelGrid grid)
    {
        int r      = grid.Resolution;
        var result = new VoxelGrid(r);

        for (int x = 0; x < r; x++)
        for (int y = 0; y < r; y++)
        for (int z = 0; z < r; z++)
        {
            if (grid.GetVoxel(x, y, z))
                result.SetVoxel(r - 1 - x, y, z, true);
        }

        return result;
    }

    /// <summary>
    /// Returns a randomly augmented copy of <paramref name="model"/>:
    /// random Y rotation (in 90° steps), optional X-flip, optional scale jitter.
    /// </summary>
    public static ProcessedModel AugmentRandom(
        ProcessedModel model,
        Random rng,
        bool allowFlip  = true,
        float maxJitter = 0.1f)
    {
        int rotations = rng.Next(4);
        bool flip     = allowFlip && rng.Next(2) == 0;

        var mesh   = model.NormalizedMesh;
        var voxels = model.Voxels;

        if (mesh != null)
        {
            mesh = RotateY(mesh, rotations * 90f);
            if (flip) mesh = FlipX(mesh);
            mesh = RandomScale(mesh, rng, maxJitter);
        }

        if (voxels != null)
        {
            voxels = RotateY90k(voxels, rotations);
            if (flip) voxels = FlipX(voxels);
        }

        return new ProcessedModel
        {
            ModelId       = model.ModelId,
            Annotation    = model.Annotation,
            FeatureVector = model.FeatureVector,
            Metadata      = model.Metadata,
            NormalizedMesh = mesh,
            Voxels         = voxels,
            // Views are view-dependent — not augmented here
            MultiViews    = model.MultiViews
        };
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    private static Mesh TransformMesh(Mesh mesh, Matrix4x4 m)
    {
        var verts   = mesh.Vertices.Select(v => Vector3.Transform(v, m)).ToArray();
        // For normals use the inverse-transpose (safe since we only use orthogonal matrices here)
        var norms   = mesh.Normals.Select(n => Vector3.Normalize(Vector3.TransformNormal(n, m))).ToArray();

        return new Mesh
        {
            Vertices = verts,
            Normals  = norms,
            UVs      = mesh.UVs,
            Indices  = mesh.Indices
        };
    }
}
