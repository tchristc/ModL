using System.Numerics;

namespace ModL.Core.Voxel;

/// <summary>
/// Represents a 3D voxel grid
/// </summary>
public class VoxelGrid
{
    private readonly bool[,,] _voxels;

    public int Resolution { get; }
    public bool[,,] Voxels => _voxels;

    public VoxelGrid(int resolution)
    {
        if (resolution <= 0 || resolution > 512)
            throw new ArgumentException("Resolution must be between 1 and 512", nameof(resolution));

        Resolution = resolution;
        _voxels = new bool[resolution, resolution, resolution];
    }

    public void SetVoxel(int x, int y, int z, bool value)
    {
        if (x < 0 || x >= Resolution || y < 0 || y >= Resolution || z < 0 || z >= Resolution)
            return;

        _voxels[x, y, z] = value;
    }

    public bool GetVoxel(int x, int y, int z)
    {
        if (x < 0 || x >= Resolution || y < 0 || y >= Resolution || z < 0 || z >= Resolution)
            return false;

        return _voxels[x, y, z];
    }

    public float[] ToFloatArray()
    {
        var array = new float[Resolution * Resolution * Resolution];
        int index = 0;

        for (int z = 0; z < Resolution; z++)
        {
            for (int y = 0; y < Resolution; y++)
            {
                for (int x = 0; x < Resolution; x++)
                {
                    array[index++] = _voxels[x, y, z] ? 1.0f : 0.0f;
                }
            }
        }

        return array;
    }

    public void FromFloatArray(float[] array, float threshold = 0.5f)
    {
        if (array.Length != Resolution * Resolution * Resolution)
            throw new ArgumentException("Array size mismatch");

        int index = 0;

        for (int z = 0; z < Resolution; z++)
        {
            for (int y = 0; y < Resolution; y++)
            {
                for (int x = 0; x < Resolution; x++)
                {
                    _voxels[x, y, z] = array[index++] >= threshold;
                }
            }
        }
    }

    public int OccupiedVoxelCount()
    {
        int count = 0;
        for (int z = 0; z < Resolution; z++)
        {
            for (int y = 0; y < Resolution; y++)
            {
                for (int x = 0; x < Resolution; x++)
                {
                    if (_voxels[x, y, z])
                        count++;
                }
            }
        }
        return count;
    }

    public void Clear()
    {
        Array.Clear(_voxels, 0, _voxels.Length);
    }

    public VoxelGrid Clone()
    {
        var clone = new VoxelGrid(Resolution);
        Array.Copy(_voxels, clone._voxels, _voxels.Length);
        return clone;
    }
}
