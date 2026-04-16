using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using ModL.Core.Geometry;

namespace ModL.Core.Rendering;

/// <summary>
/// Configuration for a single camera view
/// </summary>
public class ViewConfiguration
{
    public Vector3 CameraPosition { get; set; }
    public Vector3 LookAt { get; set; } = Vector3.Zero;
    public Vector3 Up { get; set; } = Vector3.UnitY;
    public int ImageWidth { get; set; } = 512;
    public int ImageHeight { get; set; } = 512;
    public float FieldOfView { get; set; } = 60.0f; // degrees
    public float NearPlane { get; set; } = 0.1f;
    public float FarPlane { get; set; } = 100.0f;
}

/// <summary>
/// Lighting configuration for the renderer
/// </summary>
public class LightingConfig
{
    /// <summary>Minimum scene brightness so unlit faces remain visible (0–1)</summary>
    public float Ambient { get; set; } = 0.20f;

    /// <summary>Direction the key (primary) light comes FROM, in world space</summary>
    public Vector3 KeyLightDirection { get; set; } = Vector3.Normalize(new Vector3(-1f, 2f, 1.5f));

    /// <summary>Contribution of the key light (0–1)</summary>
    public float KeyLightStrength { get; set; } = 0.60f;

    /// <summary>Direction the fill (secondary) light comes FROM, in world space</summary>
    public Vector3 FillLightDirection { get; set; } = Vector3.Normalize(new Vector3(1f, -0.5f, -1f));

    /// <summary>Contribution of the fill light (0–1)</summary>
    public float FillLightStrength { get; set; } = 0.20f;

    /// <summary>Base model colour (grey by default; tinted outputs are possible)</summary>
    public Rgb24 ModelColor { get; set; } = new Rgb24(200, 200, 210);

    /// <summary>Background colour</summary>
    public Rgb24 BackgroundColor { get; set; } = new Rgb24(245, 245, 245);

    public static LightingConfig Default => new();
}

/// <summary>
/// Renders 3D models from multiple viewpoints using a software Lambertian rasterizer.
/// Each triangle is filled with a shaded colour computed from ambient + key + fill lights.
/// Back-face culling and painter's algorithm depth sorting keep the output clean.
/// </summary>
public class MultiViewRenderer
{
    private readonly LightingConfig _lighting;

    public MultiViewRenderer(LightingConfig? lighting = null)
    {
        _lighting = lighting ?? LightingConfig.Default;
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    public Image<Rgb24>[] RenderViews(Model3D model, ViewConfiguration[] views)
    {
        var images = new Image<Rgb24>[views.Length];
        for (int i = 0; i < views.Length; i++)
            images[i] = RenderView(model, views[i]);
        return images;
    }

    public Image<Rgb24> RenderView(Model3D model, ViewConfiguration view)
    {
        var bg = new Color(_lighting.BackgroundColor);
        var image = new Image<Rgb24>(view.ImageWidth, view.ImageHeight);
        image.Mutate(ctx => ctx.Fill(bg));

        var viewMatrix = Matrix4x4.CreateLookAt(view.CameraPosition, view.LookAt, view.Up);
        var projMatrix = Matrix4x4.CreatePerspectiveFieldOfView(
            view.FieldOfView * MathF.PI / 180.0f,
            (float)view.ImageWidth / view.ImageHeight,
            view.NearPlane,
            view.FarPlane);

        var mvp = viewMatrix * projMatrix;

        foreach (var mesh in model.Meshes)
            RenderMeshShaded(image, mesh, mvp, view.ImageWidth, view.ImageHeight);

        return image;
    }

    // -----------------------------------------------------------------------
    // Camera helpers
    // -----------------------------------------------------------------------

    public ViewConfiguration[] GetStandardViews(int count = 12, float distance = 3.0f)
    {
        if (count <= 0) count = 12;

        var views = new ViewConfiguration[count];
        for (int i = 0; i < count; i++)
        {
            float angle     = 2f * MathF.PI * i / count;
            float elevation = MathF.PI / 6f; // 30° above the equator

            views[i] = new ViewConfiguration
            {
                CameraPosition = new Vector3(
                    distance * MathF.Cos(angle)  * MathF.Cos(elevation),
                    distance * MathF.Sin(elevation),
                    distance * MathF.Sin(angle)  * MathF.Cos(elevation)),
                LookAt = Vector3.Zero,
                Up     = Vector3.UnitY
            };
        }
        return views;
    }

    public ViewConfiguration[] GetOrthographicViews(float distance = 3.0f) => new[]
    {
        new ViewConfiguration { CameraPosition = new Vector3( 0,  0,  distance), Up =  Vector3.UnitY },
        new ViewConfiguration { CameraPosition = new Vector3( 0,  0, -distance), Up =  Vector3.UnitY },
        new ViewConfiguration { CameraPosition = new Vector3(-distance, 0, 0),   Up =  Vector3.UnitY },
        new ViewConfiguration { CameraPosition = new Vector3( distance, 0, 0),   Up =  Vector3.UnitY },
        new ViewConfiguration { CameraPosition = new Vector3( 0,  distance, 0),  Up =  Vector3.UnitZ },
        new ViewConfiguration { CameraPosition = new Vector3( 0, -distance, 0),  Up = -Vector3.UnitZ }
    };

    // -----------------------------------------------------------------------
    // Shaded rasterizer
    // -----------------------------------------------------------------------

    private readonly record struct ShadedTriangle(PointF A, PointF B, PointF C, float Depth, Rgb24 Color);

    private void RenderMeshShaded(Image<Rgb24> image, Mesh mesh, Matrix4x4 mvp, int width, int height)
    {
        if (mesh.Indices.Length == 0 || mesh.Vertices.Length == 0)
            return;

        // Project every vertex once
        var screen = new Vector3[mesh.Vertices.Length];
        for (int i = 0; i < mesh.Vertices.Length; i++)
            screen[i] = ProjectVertex(mesh.Vertices[i], mvp, width, height);

        bool hasNormals = mesh.Normals.Length == mesh.Vertices.Length;

        var triangles = new List<ShadedTriangle>(mesh.Indices.Length / 3);

        for (int i = 0; i < mesh.Indices.Length; i += 3)
        {
            int i0 = mesh.Indices[i];
            int i1 = mesh.Indices[i + 1];
            int i2 = mesh.Indices[i + 2];

            var s0 = screen[i0];
            var s1 = screen[i1];
            var s2 = screen[i2];

            // Discard triangles wholly outside the depth range
            if (s0.Z <= 0f || s1.Z <= 0f || s2.Z <= 0f) continue;
            if (s0.Z >= 1f && s1.Z >= 1f && s2.Z >= 1f) continue;

            // Back-face culling via screen-space winding order
            // A positive 2-D cross product means the face winds counter-clockwise
            // in screen space (Y flipped), which means it faces away from us.
            float cross = (s1.X - s0.X) * (s2.Y - s0.Y)
                        - (s1.Y - s0.Y) * (s2.X - s0.X);
            if (cross >= 0f) continue;

            // Compute or fetch the face normal in world / object space
            Vector3 normal;
            if (hasNormals)
            {
                normal = Vector3.Normalize(
                    mesh.Normals[i0] + mesh.Normals[i1] + mesh.Normals[i2]);
            }
            else
            {
                var e1 = mesh.Vertices[i1] - mesh.Vertices[i0];
                var e2 = mesh.Vertices[i2] - mesh.Vertices[i0];
                normal = Vector3.Normalize(Vector3.Cross(e1, e2));
            }

            // Lambertian shading: ambient + key diffuse + fill diffuse
            float key  = MathF.Max(0f, Vector3.Dot(normal, _lighting.KeyLightDirection));
            float fill = MathF.Max(0f, Vector3.Dot(normal, _lighting.FillLightDirection));
            float intensity = MathF.Min(1f,
                _lighting.Ambient
                + _lighting.KeyLightStrength  * key
                + _lighting.FillLightStrength * fill);

            var color = ShadeColor(_lighting.ModelColor, intensity);

            float depth = (s0.Z + s1.Z + s2.Z) / 3f;

            triangles.Add(new ShadedTriangle(
                new PointF(s0.X, s0.Y),
                new PointF(s1.X, s1.Y),
                new PointF(s2.X, s2.Y),
                depth,
                color));
        }

        // Painter's algorithm: draw far triangles first
        triangles.Sort(static (a, b) => b.Depth.CompareTo(a.Depth));

        image.Mutate(ctx =>
        {
            foreach (var tri in triangles)
            {
                var pts = new[] { tri.A, tri.B, tri.C };
                ctx.FillPolygon(new Color(tri.Color), pts);
            }
        });
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static Vector3 ProjectVertex(Vector3 vertex, Matrix4x4 mvp, int width, int height)
    {
        var clip = Vector4.Transform(new Vector4(vertex, 1f), mvp);

        if (MathF.Abs(clip.W) < 1e-4f)
            return new Vector3(-2f, -2f, -1f); // out of view

        var ndc = clip / clip.W;

        return new Vector3(
            (ndc.X + 1f) * 0.5f * width,
            (1f - ndc.Y) * 0.5f * height,  // Y is flipped in screen space
            (ndc.Z + 1f) * 0.5f);
    }

    private static Rgb24 ShadeColor(Rgb24 baseColor, float intensity)
    {
        return new Rgb24(
            (byte)MathF.Round(baseColor.R * intensity),
            (byte)MathF.Round(baseColor.G * intensity),
            (byte)MathF.Round(baseColor.B * intensity));
    }
}
