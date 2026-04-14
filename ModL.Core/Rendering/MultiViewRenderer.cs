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
/// Renders 3D models from multiple viewpoints
/// </summary>
public class MultiViewRenderer
{
    /// <summary>
    /// Renders a model from multiple standard viewpoints
    /// </summary>
    public Image<Rgb24>[] RenderViews(Model3D model, ViewConfiguration[] views)
    {
        var images = new Image<Rgb24>[views.Length];

        for (int i = 0; i < views.Length; i++)
        {
            images[i] = RenderView(model, views[i]);
        }

        return images;
    }

    /// <summary>
    /// Renders a single view of the model
    /// </summary>
    public Image<Rgb24> RenderView(Model3D model, ViewConfiguration view)
    {
        var image = new Image<Rgb24>(view.ImageWidth, view.ImageHeight);
        
        // Clear to white background
        image.Mutate(ctx => ctx.Fill(Color.White));

        // Calculate view and projection matrices
        var viewMatrix = CreateLookAtMatrix(view.CameraPosition, view.LookAt, view.Up);
        var projMatrix = CreatePerspectiveMatrix(
            view.FieldOfView * MathF.PI / 180.0f,
            (float)view.ImageWidth / view.ImageHeight,
            view.NearPlane,
            view.FarPlane);

        // Simple wireframe rendering
        foreach (var mesh in model.Meshes)
        {
            RenderMeshWireframe(image, mesh, viewMatrix, projMatrix, view.ImageWidth, view.ImageHeight);
        }

        return image;
    }

    private void RenderMeshWireframe(
        Image<Rgb24> image,
        Mesh mesh,
        Matrix4x4 viewMatrix,
        Matrix4x4 projMatrix,
        int width,
        int height)
    {
        var mvp = viewMatrix * projMatrix;
        var projectedVertices = new Vector3[mesh.Vertices.Length];

        // Project all vertices
        for (int i = 0; i < mesh.Vertices.Length; i++)
        {
            projectedVertices[i] = ProjectVertex(mesh.Vertices[i], mvp, width, height);
        }

        // Draw edges
        image.Mutate(ctx =>
        {
            for (int i = 0; i < mesh.Indices.Length; i += 3)
            {
                var v0 = projectedVertices[mesh.Indices[i]];
                var v1 = projectedVertices[mesh.Indices[i + 1]];
                var v2 = projectedVertices[mesh.Indices[i + 2]];

                // Check if triangle is in view
                if (v0.Z > 0 && v0.Z < 1 && v1.Z > 0 && v1.Z < 1 && v2.Z > 0 && v2.Z < 1)
                {
                    DrawLine(ctx, v0, v1, Color.Black);
                    DrawLine(ctx, v1, v2, Color.Black);
                    DrawLine(ctx, v2, v0, Color.Black);
                }
            }
        });
    }

    private Vector3 ProjectVertex(Vector3 vertex, Matrix4x4 mvp, int width, int height)
    {
        var transformed = Vector4.Transform(new Vector4(vertex, 1.0f), mvp);
        
        if (MathF.Abs(transformed.W) < 0.0001f)
            return new Vector3(-1, -1, -1);

        transformed /= transformed.W;

        // NDC to screen space
        var screenX = (transformed.X + 1) * 0.5f * width;
        var screenY = (1 - transformed.Y) * 0.5f * height;
        var depth = (transformed.Z + 1) * 0.5f;

        return new Vector3(screenX, screenY, depth);
    }

    private void DrawLine(IImageProcessingContext ctx, Vector3 start, Vector3 end, Color color)
    {
        var p1 = new PointF(start.X, start.Y);
        var p2 = new PointF(end.X, end.Y);
        
        ctx.DrawLine(color, 1.0f, p1, p2);
    }

    /// <summary>
    /// Generates standard camera positions around the model
    /// </summary>
    public ViewConfiguration[] GetStandardViews(int count = 12, float distance = 3.0f)
    {
        var views = new List<ViewConfiguration>();

        if (count <= 0)
            count = 12;

        // Generate views around the model in a circle
        for (int i = 0; i < count; i++)
        {
            float angle = (float)(2 * Math.PI * i / count);
            float elevation = MathF.PI / 6; // 30 degrees up

            var cameraPos = new Vector3(
                distance * MathF.Cos(angle) * MathF.Cos(elevation),
                distance * MathF.Sin(elevation),
                distance * MathF.Sin(angle) * MathF.Cos(elevation)
            );

            views.Add(new ViewConfiguration
            {
                CameraPosition = cameraPos,
                LookAt = Vector3.Zero,
                Up = Vector3.UnitY
            });
        }

        return views.ToArray();
    }

    /// <summary>
    /// Generates 6 orthographic views (front, back, left, right, top, bottom)
    /// </summary>
    public ViewConfiguration[] GetOrthographicViews(float distance = 3.0f)
    {
        return new[]
        {
            // Front
            new ViewConfiguration { CameraPosition = new Vector3(0, 0, distance), Up = Vector3.UnitY },
            // Back
            new ViewConfiguration { CameraPosition = new Vector3(0, 0, -distance), Up = Vector3.UnitY },
            // Left
            new ViewConfiguration { CameraPosition = new Vector3(-distance, 0, 0), Up = Vector3.UnitY },
            // Right
            new ViewConfiguration { CameraPosition = new Vector3(distance, 0, 0), Up = Vector3.UnitY },
            // Top
            new ViewConfiguration { CameraPosition = new Vector3(0, distance, 0), Up = Vector3.UnitZ },
            // Bottom
            new ViewConfiguration { CameraPosition = new Vector3(0, -distance, 0), Up = -Vector3.UnitZ }
        };
    }

    private Matrix4x4 CreateLookAtMatrix(Vector3 cameraPosition, Vector3 target, Vector3 up)
    {
        return Matrix4x4.CreateLookAt(cameraPosition, target, up);
    }

    private Matrix4x4 CreatePerspectiveMatrix(float fov, float aspectRatio, float nearPlane, float farPlane)
    {
        return Matrix4x4.CreatePerspectiveFieldOfView(fov, aspectRatio, nearPlane, farPlane);
    }
}
