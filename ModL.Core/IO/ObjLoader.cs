using System.Globalization;
using System.Numerics;
using ModL.Core.Geometry;

namespace ModL.Core.IO;

/// <summary>
/// Loads Wavefront OBJ files
/// </summary>
public class ObjLoader : IModelLoader
{
    public string[] SupportedExtensions => new[] { ".obj" };

    public bool CanLoad(string extension)
    {
        return SupportedExtensions.Contains(extension.ToLowerInvariant());
    }

    public Model3D Load(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"OBJ file not found: {filePath}");

        var vertices = new List<Vector3>();
        var normals = new List<Vector3>();
        var uvs = new List<Vector2>();
        var faces = new List<Face>();
        var materials = new Dictionary<string, Material>();
        var meshGroups = new Dictionary<string, List<Face>>();
        
        string currentGroup = "default";
        string currentMaterial = "";
        string? mtlFile = null;

        var lines = File.ReadAllLines(filePath);
        
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#'))
                continue;

            var parts = trimmed.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
                continue;

            switch (parts[0].ToLower())
            {
                case "v": // Vertex
                    if (parts.Length >= 4)
                    {
                        vertices.Add(new Vector3(
                            ParseFloat(parts[1]),
                            ParseFloat(parts[2]),
                            ParseFloat(parts[3])));
                    }
                    break;

                case "vn": // Normal
                    if (parts.Length >= 4)
                    {
                        normals.Add(new Vector3(
                            ParseFloat(parts[1]),
                            ParseFloat(parts[2]),
                            ParseFloat(parts[3])));
                    }
                    break;

                case "vt": // Texture coordinate
                    if (parts.Length >= 3)
                    {
                        uvs.Add(new Vector2(
                            ParseFloat(parts[1]),
                            ParseFloat(parts[2])));
                    }
                    break;

                case "f": // Face
                    if (parts.Length >= 4)
                    {
                        var face = ParseFace(parts.Skip(1).ToArray(), vertices.Count, normals.Count, uvs.Count);
                        face.Material = currentMaterial;
                        faces.Add(face);

                        if (!meshGroups.ContainsKey(currentGroup))
                            meshGroups[currentGroup] = new List<Face>();
                        meshGroups[currentGroup].Add(face);
                    }
                    break;

                case "g": // Group
                case "o": // Object
                    if (parts.Length >= 2)
                        currentGroup = parts[1];
                    break;

                case "usemtl": // Use material
                    if (parts.Length >= 2)
                        currentMaterial = parts[1];
                    break;

                case "mtllib": // Material library
                    if (parts.Length >= 2)
                        mtlFile = Path.Combine(Path.GetDirectoryName(filePath) ?? "", parts[1]);
                    break;
            }
        }

        // Load materials if MTL file is specified
        if (mtlFile != null && File.Exists(mtlFile))
        {
            materials = LoadMtlFile(mtlFile);
        }

        // Build meshes from face groups
        var meshes = new List<Mesh>();
        var materialList = new List<Material>();
        var materialIndices = new Dictionary<string, int>();

        foreach (var group in meshGroups)
        {
            var mesh = BuildMesh(group.Value, vertices, normals, uvs, materials, materialList, materialIndices);
            mesh.Name = group.Key;
            meshes.Add(mesh);
        }

        var model = new Model3D
        {
            Name = Path.GetFileNameWithoutExtension(filePath),
            Meshes = meshes.ToArray(),
            Materials = materialList.ToArray()
        };

        model.CalculateBoundingBox();
        return model;
    }

    private static Mesh BuildMesh(
        List<Face> faces,
        List<Vector3> vertices,
        List<Vector3> normals,
        List<Vector2> uvs,
        Dictionary<string, Material> materials,
        List<Material> materialList,
        Dictionary<string, int> materialIndices)
    {
        var meshVertices = new List<Vector3>();
        var meshNormals = new List<Vector3>();
        var meshUVs = new List<Vector2>();
        var meshIndices = new List<int>();
        var vertexCache = new Dictionary<string, int>();

        int materialIndex = -1;
        if (faces.Count > 0 && !string.IsNullOrEmpty(faces[0].Material))
        {
            var matName = faces[0].Material;
            if (!materialIndices.ContainsKey(matName))
            {
                if (materials.TryGetValue(matName, out var material))
                {
                    materialList.Add(material);
                }
                else
                {
                    materialList.Add(new Material { Name = matName });
                }
                materialIndices[matName] = materialList.Count - 1;
            }
            materialIndex = materialIndices[matName];
        }

        foreach (var face in faces)
        {
            for (int i = 0; i < face.Vertices.Count; i++)
            {
                var fv = face.Vertices[i];
                var key = $"{fv.VertexIndex}_{fv.NormalIndex}_{fv.UVIndex}";

                if (!vertexCache.TryGetValue(key, out var index))
                {
                    index = meshVertices.Count;
                    vertexCache[key] = index;

                    meshVertices.Add(vertices[fv.VertexIndex]);
                    
                    if (fv.NormalIndex >= 0 && fv.NormalIndex < normals.Count)
                        meshNormals.Add(normals[fv.NormalIndex]);
                    else
                        meshNormals.Add(Vector3.UnitY);

                    if (fv.UVIndex >= 0 && fv.UVIndex < uvs.Count)
                        meshUVs.Add(uvs[fv.UVIndex]);
                    else
                        meshUVs.Add(Vector2.Zero);
                }

                meshIndices.Add(index);
            }
        }

        var mesh = new Mesh
        {
            Vertices = meshVertices.ToArray(),
            Normals = meshNormals.ToArray(),
            UVs = meshUVs.ToArray(),
            Indices = meshIndices.ToArray(),
            MaterialIndex = materialIndex
        };

        // Calculate normals if not present
        if (meshNormals.Count == 0 || meshNormals.All(n => n == Vector3.UnitY))
        {
            mesh.CalculateNormals();
        }

        return mesh;
    }

    private static Face ParseFace(string[] parts, int vertexCount, int normalCount, int uvCount)
    {
        var face = new Face();

        foreach (var part in parts)
        {
            var indices = part.Split('/');
            var fv = new FaceVertex();

            if (indices.Length >= 1 && !string.IsNullOrEmpty(indices[0]))
            {
                var vi = int.Parse(indices[0]);
                fv.VertexIndex = vi > 0 ? vi - 1 : vertexCount + vi;
            }

            if (indices.Length >= 2 && !string.IsNullOrEmpty(indices[1]))
            {
                var ti = int.Parse(indices[1]);
                fv.UVIndex = ti > 0 ? ti - 1 : uvCount + ti;
            }

            if (indices.Length >= 3 && !string.IsNullOrEmpty(indices[2]))
            {
                var ni = int.Parse(indices[2]);
                fv.NormalIndex = ni > 0 ? ni - 1 : normalCount + ni;
            }

            face.Vertices.Add(fv);
        }

        return face;
    }

    private static Dictionary<string, Material> LoadMtlFile(string mtlPath)
    {
        var materials = new Dictionary<string, Material>();
        Material? currentMaterial = null;
        var basePath = Path.GetDirectoryName(mtlPath) ?? "";

        var lines = File.ReadAllLines(mtlPath);

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#'))
                continue;

            var parts = trimmed.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
                continue;

            switch (parts[0].ToLower())
            {
                case "newmtl":
                    if (parts.Length >= 2)
                    {
                        currentMaterial = new Material { Name = parts[1] };
                        materials[parts[1]] = currentMaterial;
                    }
                    break;

                case "kd": // Diffuse color
                    if (currentMaterial != null && parts.Length >= 4)
                    {
                        currentMaterial.DiffuseColor = new Vector4(
                            ParseFloat(parts[1]),
                            ParseFloat(parts[2]),
                            ParseFloat(parts[3]),
                            1.0f);
                    }
                    break;

                case "map_kd": // Diffuse texture
                    if (currentMaterial != null && parts.Length >= 2)
                    {
                        currentMaterial.DiffuseTexture = Path.Combine(basePath, parts[1]);
                    }
                    break;

                case "map_bump":
                case "bump":
                    if (currentMaterial != null && parts.Length >= 2)
                    {
                        currentMaterial.NormalMap = Path.Combine(basePath, parts[1]);
                    }
                    break;
            }
        }

        return materials;
    }

    private static float ParseFloat(string value)
    {
        return float.Parse(value, CultureInfo.InvariantCulture);
    }

    private class Face
    {
        public List<FaceVertex> Vertices { get; } = new();
        public string Material { get; set; } = "";
    }

    private struct FaceVertex
    {
        public int VertexIndex;
        public int NormalIndex;
        public int UVIndex;

        public FaceVertex()
        {
            VertexIndex = -1;
            NormalIndex = -1;
            UVIndex = -1;
        }
    }
}
