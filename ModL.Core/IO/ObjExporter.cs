using System.Globalization;
using System.Text;
using ModL.Core.Geometry;

namespace ModL.Core.IO;

/// <summary>
/// Exports models to Wavefront OBJ format
/// </summary>
public class ObjExporter : IModelExporter
{
    public string[] SupportedExtensions => new[] { ".obj" };

    public void Export(Model3D model, string filePath)
    {
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var objContent = new StringBuilder();
        var mtlContent = new StringBuilder();
        var mtlFileName = Path.GetFileNameWithoutExtension(filePath) + ".mtl";
        var hasMaterials = model.Materials.Length > 0;

        // Write header
        objContent.AppendLine("# Exported by ModL");
        objContent.AppendLine($"# Model: {model.Name}");
        objContent.AppendLine();

        if (hasMaterials)
        {
            objContent.AppendLine($"mtllib {mtlFileName}");
            objContent.AppendLine();
        }

        int vertexOffset = 1; // OBJ indices are 1-based

        foreach (var mesh in model.Meshes)
        {
            objContent.AppendLine($"o {mesh.Name}");
            objContent.AppendLine();

            // Write vertices
            foreach (var vertex in mesh.Vertices)
            {
                objContent.AppendLine($"v {FormatFloat(vertex.X)} {FormatFloat(vertex.Y)} {FormatFloat(vertex.Z)}");
            }
            objContent.AppendLine();

            // Write normals
            if (mesh.Normals.Length > 0)
            {
                foreach (var normal in mesh.Normals)
                {
                    objContent.AppendLine($"vn {FormatFloat(normal.X)} {FormatFloat(normal.Y)} {FormatFloat(normal.Z)}");
                }
                objContent.AppendLine();
            }

            // Write UVs
            if (mesh.UVs.Length > 0)
            {
                foreach (var uv in mesh.UVs)
                {
                    objContent.AppendLine($"vt {FormatFloat(uv.X)} {FormatFloat(uv.Y)}");
                }
                objContent.AppendLine();
            }

            // Write material
            if (mesh.MaterialIndex >= 0 && mesh.MaterialIndex < model.Materials.Length)
            {
                objContent.AppendLine($"usemtl {model.Materials[mesh.MaterialIndex].Name}");
            }

            // Write faces
            bool hasNormals = mesh.Normals.Length > 0;
            bool hasUVs = mesh.UVs.Length > 0;

            for (int i = 0; i < mesh.Indices.Length; i += 3)
            {
                var i0 = mesh.Indices[i] + vertexOffset;
                var i1 = mesh.Indices[i + 1] + vertexOffset;
                var i2 = mesh.Indices[i + 2] + vertexOffset;

                if (hasNormals && hasUVs)
                {
                    objContent.AppendLine($"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}");
                }
                else if (hasUVs)
                {
                    objContent.AppendLine($"f {i0}/{i0} {i1}/{i1} {i2}/{i2}");
                }
                else if (hasNormals)
                {
                    objContent.AppendLine($"f {i0}//{i0} {i1}//{i1} {i2}//{i2}");
                }
                else
                {
                    objContent.AppendLine($"f {i0} {i1} {i2}");
                }
            }

            objContent.AppendLine();
            vertexOffset += mesh.Vertices.Length;
        }

        // Write MTL file if materials exist
        if (hasMaterials)
        {
            mtlContent.AppendLine("# Exported by ModL");
            mtlContent.AppendLine();

            foreach (var material in model.Materials)
            {
                mtlContent.AppendLine($"newmtl {material.Name}");
                mtlContent.AppendLine($"Ka {FormatFloat(material.DiffuseColor.X)} {FormatFloat(material.DiffuseColor.Y)} {FormatFloat(material.DiffuseColor.Z)}");
                mtlContent.AppendLine($"Kd {FormatFloat(material.DiffuseColor.X)} {FormatFloat(material.DiffuseColor.Y)} {FormatFloat(material.DiffuseColor.Z)}");
                mtlContent.AppendLine($"Ks 0.5 0.5 0.5");
                mtlContent.AppendLine($"Ns {FormatFloat((1.0f - material.Roughness) * 1000)}");
                mtlContent.AppendLine($"d {FormatFloat(material.DiffuseColor.W)}");

                if (!string.IsNullOrEmpty(material.DiffuseTexture))
                {
                    var textureName = Path.GetFileName(material.DiffuseTexture);
                    mtlContent.AppendLine($"map_Kd {textureName}");
                }

                if (!string.IsNullOrEmpty(material.NormalMap))
                {
                    var normalName = Path.GetFileName(material.NormalMap);
                    mtlContent.AppendLine($"map_Bump {normalName}");
                }

                mtlContent.AppendLine();
            }

            var mtlPath = Path.Combine(directory ?? "", mtlFileName);
            File.WriteAllText(mtlPath, mtlContent.ToString());
        }

        // Write OBJ file
        File.WriteAllText(filePath, objContent.ToString());
    }

    private static string FormatFloat(float value)
    {
        return value.ToString("F6", CultureInfo.InvariantCulture);
    }
}
