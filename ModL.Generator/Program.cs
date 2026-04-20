using CommandLine;
using Spectre.Console;
using Serilog;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using ModL.ML.Models;
using ModL.ML.Training;
using ModL.ML.Data;
using ModL.Data.Pipeline;
using ModL.Core.IO;
using static TorchSharp.torch;

namespace ModL.Generator;

class Program
{
    static async Task<int> Main(string[] args)
    {
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Information()
            .WriteTo.Console()
            .WriteTo.File("logs/modl-gen-.log", rollingInterval: RollingInterval.Day)
            .CreateLogger();

        try
        {
            AnsiConsole.Write(
                new FigletText("ModL Generator")
                    .LeftJustified()
                    .Color(Spectre.Console.Color.Cyan1));
            AnsiConsole.MarkupLine("[dim]3D Model AI Generation System[/]");
            AnsiConsole.WriteLine();

            return await Parser.Default.ParseArguments<
                GenerateCommand,
                RetrieveCommand,
                InspectEmbeddingCommand>(args)
                .MapResult(
                    (GenerateCommand cmd)          => cmd.ExecuteAsync(),
                    (RetrieveCommand cmd)          => cmd.ExecuteAsync(),
                    (InspectEmbeddingCommand cmd)  => cmd.ExecuteAsync(),
                    errs => Task.FromResult(1));
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Unhandled exception");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return 1;
        }
        finally
        {
            Log.CloseAndFlush();
        }
    }
}

// ---------------------------------------------------------------------------
// generate command
// ---------------------------------------------------------------------------

[Verb("generate", HelpText = "Find nearest models to a text prompt via embedding space lookup")]
class GenerateCommand
{
    [Value(0, Required = true, HelpText = "Text prompt (e.g. \"a modern chair\")")]
    public string Prompt { get; set; } = string.Empty;

    [Option("model", Required = true, HelpText = "Path to trained checkpoint directory")]
    public string ModelPath { get; set; } = string.Empty;

    [Option("embeddings", Required = true, HelpText = "Path to embeddings JSON (from modl export)")]
    public string EmbeddingsJson { get; set; } = string.Empty;

    [Option("processed", Required = true, HelpText = "Processed data directory")]
    public string ProcessedDir { get; set; } = string.Empty;

    [Option("top", Default = 5, HelpText = "Number of nearest neighbours to return")]
    public int TopK { get; set; }

    [Option("config", HelpText = "Training config JSON")]
    public string? ConfigFile { get; set; }

    public async Task<int> ExecuteAsync()
    {
        try
        {
            AnsiConsole.MarkupLine($"[cyan]Prompt:[/] [bold]{Prompt}[/]");

            var cfg = ConfigFile != null && File.Exists(ConfigFile)
                ? TrainingConfig.FromJson(ConfigFile)
                : new TrainingConfig { ProcessedDir = ProcessedDir };

            var model = new ModLModel(cfg.NumClasses, cfg.VoxelLatentDim, cfg.ViewLatentDim, cfg.EmbeddingDim);
            model.Load(ModelPath);
            model.Eval();

            var index   = EmbeddingIndex.Load(EmbeddingsJson);
            var seedDir = FindSeedByKeyword(ProcessedDir, Prompt);

            if (seedDir == null)
            {
                AnsiConsole.MarkupLine("[yellow]No seed model matched the prompt keywords. Try a category name.[/]");
                AnsiConsole.MarkupLine($"[dim]Available model IDs: {index.Count} total[/]");
                model.Dispose();
                return 0;
            }

            AnsiConsole.MarkupLine($"[dim]Seed model: {Path.GetFileName(seedDir)}[/]");

            var batchCfg = new TrainingBatchConfig
            {
                ProcessedDir    = ProcessedDir,
                VoxelResolution = cfg.VoxelResolution,
                NumViews        = cfg.NumViews,
                ViewImageSize   = cfg.ViewImageSize,
                BatchSize       = 1
            };
            var loader  = new ModelDataLoader(batchCfg);
            var store   = new ProcessedModelStore();
            var seedRec = store.Load(seedDir, loadViews: true);

            float[] queryVec;
            using (var noGrad = no_grad())
            {
                var singleBatch = BuildSingleBatch(seedRec, batchCfg, loader.LabelMap);
                using (singleBatch)
                using (var emb = model.Embed(singleBatch.Voxels, singleBatch.Views))
                using (var cpu = emb.cpu())
                {
                    int d = (int)cpu.shape[1];
                    queryVec = new float[d];
                    for (int j = 0; j < d; j++)
                        queryVec[j] = cpu[0, j].item<float>();
                }
            }

            var neighbours = index.Search(queryVec, TopK);

            var table = new Table()
                .AddColumn("Rank")
                .AddColumn("Model ID")
                .AddColumn(new TableColumn("Cosine Sim").RightAligned());

            int rank = 1;
            foreach (var (id, sim) in neighbours)
                table.AddRow(rank++.ToString(), id, $"{sim:F4}");

            AnsiConsole.MarkupLine($"\n[green]Top-{TopK} nearest models:[/]");
            AnsiConsole.Write(table);

            model.Dispose();
            return 0;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Generate failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return 1;
        }
    }

    private static string? FindSeedByKeyword(string processedDir, string prompt)
    {
        var keywords = prompt.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var store    = new ProcessedModelStore();
        foreach (var dir in ProcessedModelStore.ListModelDirs(processedDir))
        {
            try
            {
                var rec = store.Load(dir, loadViews: false);
                var cat = rec.Annotation?.Category?.ToLowerInvariant() ?? string.Empty;
                var id  = rec.ModelId.ToLowerInvariant();
                if (keywords.Any(k => cat.Contains(k) || id.Contains(k)))
                    return dir;
            }
            catch { }
        }
        return null;
    }

    private static ModelBatch BuildSingleBatch(
        ProcessedModel rec,
        TrainingBatchConfig cfg,
        IReadOnlyDictionary<string, int> labelMap)
    {
        int r = cfg.VoxelResolution, v = cfg.NumViews, h = cfg.ViewImageSize, w = cfg.ViewImageSize;
        var voxelData = rec.Voxels?.ToFloatArray() ?? new float[r * r * r];
        var viewData  = new float[v * 3 * h * w];

        if (rec.MultiViews != null)
        {
            for (int vi = 0; vi < Math.Min(rec.MultiViews.Length, v); vi++)
            {
                var img     = rec.MultiViews[vi];
                var resized = img.Width == w && img.Height == h
                    ? img
                    : img.Clone(ctx => ctx.Resize(new ResizeOptions { Size = new SixLabors.ImageSharp.Size(w, h) }));

                for (int y = 0; y < h; y++)
                {
                    var row = resized.Frames.RootFrame.PixelBuffer.DangerousGetRowSpan(y);
                    for (int x = 0; x < w; x++)
                    {
                        var px    = row[x];
                        int pBase = vi * 3 * h * w + y * w + x;
                        viewData[pBase]             = px.R / 255f;
                        viewData[pBase + h * w]     = px.G / 255f;
                        viewData[pBase + 2 * h * w] = px.B / 255f;
                    }
                }
            }
        }

        var cat   = rec.Annotation?.Category ?? "unknown";
        var label = labelMap.TryGetValue(cat, out int l) ? l : 0;

        return new ModelBatch(
            tensor(voxelData).reshape(1, 1, r, r, r),
            tensor(viewData).reshape(1, v, 3, h, w),
            tensor(new long[] { label }));
    }
}

// ---------------------------------------------------------------------------
// retrieve command
// ---------------------------------------------------------------------------

[Verb("retrieve", HelpText = "Find K most similar models to a given model ID")]
class RetrieveCommand
{
    [Value(0, Required = true, HelpText = "Model ID to use as query")]
    public string QueryModelId { get; set; } = string.Empty;

    [Option("embeddings", Required = true, HelpText = "Path to embeddings JSON")]
    public string EmbeddingsJson { get; set; } = string.Empty;

    [Option("top", Default = 10, HelpText = "Number of nearest neighbours")]
    public int TopK { get; set; }

    public Task<int> ExecuteAsync()
    {
        try
        {
            var index = EmbeddingIndex.Load(EmbeddingsJson);

            if (!index.TryGet(QueryModelId, out var queryVec))
            {
                AnsiConsole.MarkupLine($"[red]Model ID '{QueryModelId}' not found in embeddings.[/]");
                return Task.FromResult(1);
            }

            var neighbours = index.Search(queryVec!, TopK + 1)
                .Where(n => n.Id != QueryModelId)
                .Take(TopK)
                .ToList();

            var table = new Table()
                .AddColumn("Rank")
                .AddColumn("Model ID")
                .AddColumn(new TableColumn("Cosine Sim").RightAligned());

            int rank = 1;
            foreach (var (id, sim) in neighbours)
                table.AddRow(rank++.ToString(), id, $"{sim:F4}");

            AnsiConsole.MarkupLine($"[green]Nearest to [bold]{QueryModelId}[/]:[/]");
            AnsiConsole.Write(table);
            return Task.FromResult(0);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Retrieve failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return Task.FromResult(1);
        }
    }
}

// ---------------------------------------------------------------------------
// inspect-embedding command
// ---------------------------------------------------------------------------

[Verb("inspect-embedding", HelpText = "Show statistics about a saved embeddings file")]
class InspectEmbeddingCommand
{
    [Value(0, Required = true, HelpText = "Path to embeddings JSON")]
    public string EmbeddingsJson { get; set; } = string.Empty;

    public Task<int> ExecuteAsync()
    {
        try
        {
            var index = EmbeddingIndex.Load(EmbeddingsJson);
            AnsiConsole.MarkupLine($"[green]Embedding index:[/] {EmbeddingsJson}");
            AnsiConsole.MarkupLine($"  Models : [yellow]{index.Count}[/]");
            AnsiConsole.MarkupLine($"  Dim    : [yellow]{index.Dimension}[/]");
            return Task.FromResult(0);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Inspect failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return Task.FromResult(1);
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingIndex — in-memory cosine similarity search
// ---------------------------------------------------------------------------

public sealed class EmbeddingIndex
{
    private readonly Dictionary<string, float[]> _data;

    public int Count     => _data.Count;
    public int Dimension => _data.Count > 0 ? _data.Values.First().Length : 0;

    private EmbeddingIndex(Dictionary<string, float[]> data) => _data = data;

    public static EmbeddingIndex Load(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath);
        var raw  = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, float[]>>(json)
                   ?? throw new InvalidDataException("Cannot parse embeddings JSON: " + jsonPath);
        return new EmbeddingIndex(raw);
    }

    public bool TryGet(string modelId, out float[]? vec)
        => _data.TryGetValue(modelId, out vec);

    public IReadOnlyList<(string Id, float Similarity)> Search(float[] query, int k)
    {
        float qNorm = Norm(query);
        if (qNorm < 1e-8f) return Array.Empty<(string, float)>();

        return _data
            .Select(kv =>
            {
                float sim = Norm(kv.Value) is float vn && vn > 1e-8f
                    ? Dot(query, kv.Value) / (qNorm * vn)
                    : 0f;
                return (Id: kv.Key, Similarity: sim);
            })
            .OrderByDescending(x => x.Similarity)
            .Take(k)
            .ToList();
    }

    private static float Dot(float[] a, float[] b)
    {
        float s = 0f;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    private static float Norm(float[] v)
    {
        float s = 0f;
        for (int i = 0; i < v.Length; i++) s += v[i] * v[i];
        return MathF.Sqrt(s);
    }
}

