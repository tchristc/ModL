using ModL.Core.Geometry;

namespace ModL.Core.IO;

/// <summary>
/// Loads FBX files using Assimp.NET
/// </summary>
public class FbxLoader : IModelLoader
{
    public string[] SupportedExtensions => new[] { ".fbx", ".dae", ".gltf", ".glb", ".stl", ".ply" };

    public bool CanLoad(string extension)
    {
        return SupportedExtensions.Contains(extension.ToLowerInvariant());
    }

    public Model3D Load(string filePath)
    {
        // TODO: Implement using Assimp.NET when needed
        // For now, this is a placeholder
        throw new NotImplementedException(
            "FBX loader not yet implemented. Use AssimpNet integration for production use.");
    }
}

/// <summary>
/// Exports to FBX format
/// </summary>
public class FbxExporter : IModelExporter
{
    public string[] SupportedExtensions => new[] { ".fbx" };

    public void Export(Model3D model, string filePath)
    {
        // TODO: Implement using Assimp.NET or custom FBX writer
        throw new NotImplementedException(
            "FBX exporter not yet implemented. Use OBJ format for now.");
    }
}
