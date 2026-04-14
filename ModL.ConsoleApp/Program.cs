using CommandLine;
using Spectre.Console;
using Serilog;
using ModL.Data.Datasets;
using ModL.Data.Pipeline;
using ModL.Core.IO;
using ModL.Data.Annotations;
using SixLabors.ImageSharp.Formats.Png;

namespace ModL.ConsoleApp;

class Program
{
    static async Task<int> Main(string[] args)
    {
        // Configure logging
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Information()
            .WriteTo.Console()
            .WriteTo.File("logs/modl-.log", rollingInterval: RollingInterval.Day)
            .CreateLogger();

        try
        {
            AnsiConsole.Write(
                new FigletText("ModL Training")
                    .LeftJustified()
                    .Color(Color.Green));

            AnsiConsole.MarkupLine("[dim]3D Model AI Training System[/]");
            AnsiConsole.WriteLine();

            return await Parser.Default.ParseArguments<
                DownloadCommand,
                PreprocessCommand,
                TrainCommand,
                EvaluateCommand,
                ExportCommand>(args)
                .MapResult(
                    (DownloadCommand cmd) => cmd.ExecuteAsync(),
                    (PreprocessCommand cmd) => cmd.ExecuteAsync(),
                    (TrainCommand cmd) => cmd.ExecuteAsync(),
                    (EvaluateCommand cmd) => cmd.ExecuteAsync(),
                    (ExportCommand cmd) => cmd.ExecuteAsync(),
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

[Verb("download", HelpText = "Download a 3D model dataset")]
class DownloadCommand
{
    [Value(0, Required = true, HelpText = "Dataset name (shapenet, modelnet)")]
    public string Dataset { get; set; } = string.Empty;

    [Value(1, Required = true, HelpText = "Output directory path")]
    public string OutputPath { get; set; } = string.Empty;

    [Option("categories", HelpText = "Specific categories to download (comma-separated)")]
    public string? Categories { get; set; }

    public async Task<int> ExecuteAsync()
    {
        try
        {
            AnsiConsole.MarkupLine($"[green]Downloading {Dataset} dataset...[/]");

            var config = new DatasetConfig
            {
                Name = Dataset,
                LocalPath = OutputPath,
                Categories = Categories?.Split(',') ?? Array.Empty<string>()
            };

            var downloader = DatasetDownloaderFactory.Create(Dataset);

            await AnsiConsole.Progress()
                .StartAsync(async ctx =>
                {
                    var task = ctx.AddTask($"[green]Downloading {Dataset}[/]");

                    var progress = new Progress<DownloadProgress>(p =>
                    {
                        task.Description = p.Message;
                        if (p.TotalBytes > 0)
                        {
                            task.Value = p.PercentComplete;
                        }
                    });

                    await downloader.DownloadAsync(config, progress);
                    task.Value = 100;
                });

            AnsiConsole.MarkupLine("[green]✓ Download complete![/]");
            return 0;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Download failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return 1;
        }
    }
}

[Verb("preprocess", HelpText = "Preprocess 3D models for training")]
class PreprocessCommand
{
    [Value(0, Required = true, HelpText = "Input directory with raw models")]
    public string InputDir { get; set; } = string.Empty;

    [Value(1, Required = true, HelpText = "Output directory for processed data")]
    public string OutputDir { get; set; } = string.Empty;

    [Option("voxel-size", Default = 64, HelpText = "Voxel grid resolution (32, 64, 128, 256)")]
    public int VoxelSize { get; set; }

    [Option("views", Default = 12, HelpText = "Number of multi-view renderings")]
    public int Views { get; set; }

    [Option("parallel", Default = -1, HelpText = "Number of parallel threads (-1 for auto)")]
    public int Parallel { get; set; }

    public async Task<int> ExecuteAsync()
    {
        try
        {
            AnsiConsole.MarkupLine($"[green]Preprocessing models from {InputDir}...[/]");

            if (!Directory.Exists(InputDir))
            {
                AnsiConsole.MarkupLine("[red]Input directory does not exist![/]");
                return 1;
            }

            Directory.CreateDirectory(OutputDir);

            var config = new PreprocessingConfig
            {
                VoxelResolution = VoxelSize,
                MultiViewCount = Views,
                Normalize = true,
                CalculateNormals = true
            };

            // Find all model files
            var modelFiles = Directory.GetFiles(InputDir, "*.obj", SearchOption.AllDirectories)
                .Concat(Directory.GetFiles(InputDir, "*.fbx", SearchOption.AllDirectories))
                .ToArray();

            AnsiConsole.MarkupLine($"Found [yellow]{modelFiles.Length}[/] models");

            var processor = new DataProcessor();
            var processed = 0;

            await AnsiConsole.Progress()
                .StartAsync(async ctx =>
                {
                    var task = ctx.AddTask("[green]Processing models[/]", maxValue: modelFiles.Length);

                    foreach (var modelFile in modelFiles)
                    {
                        try
                        {
                            task.Description = $"[green]Processing {Path.GetFileName(modelFile)}[/]";

                            var model = ModelIOFactory.LoadModel(modelFile);
                            var annotation = TryLoadAnnotation(modelFile);

                            var result = processor.Process(model, annotation, config);

                            // Save processed data
                            var outputName = Path.GetFileNameWithoutExtension(modelFile);
                            SaveProcessedModel(result, Path.Combine(OutputDir, outputName));

                            processed++;
                            task.Increment(1);
                        }
                        catch (Exception ex)
                        {
                            Log.Warning(ex, "Failed to process {File}", modelFile);
                        }
                    }
                });

            AnsiConsole.MarkupLine($"[green]✓ Processed {processed}/{modelFiles.Length} models![/]");
            return 0;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Preprocessing failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return 1;
        }
    }

    private ModelAnnotation? TryLoadAnnotation(string modelFile)
    {
        var annotationFile = Path.ChangeExtension(modelFile, ".json");
        if (File.Exists(annotationFile))
        {
            return AnnotationParserFactory.Parse(annotationFile);
        }
        return null;
    }

    private void SaveProcessedModel(ProcessedModel model, string outputPath)
    {
        Directory.CreateDirectory(outputPath);

        // Save voxel grid
        if (model.Voxels != null)
        {
            var voxelData = model.Voxels.ToFloatArray();
            var voxelPath = Path.Combine(outputPath, "voxels.bin");
            using var fs = File.Create(voxelPath);
            using var bw = new BinaryWriter(fs);
            bw.Write(model.Voxels.Resolution);
            foreach (var v in voxelData)
                bw.Write(v);
        }

        // Save multi-view images
        if (model.MultiViews != null)
        {
            var viewsDir = Path.Combine(outputPath, "views");
            Directory.CreateDirectory(viewsDir);
            for (int i = 0; i < model.MultiViews.Length; i++)
            {
                model.MultiViews[i].Save(Path.Combine(viewsDir, $"view_{i:D2}.png"));
            }
        }

        // Save metadata
        var metadataPath = Path.Combine(outputPath, "metadata.json");
        var metadata = new
        {
            model.ModelId,
            model.Annotation?.Category,
            model.Annotation?.Tags,
            model.Metadata,
            FeatureVector = model.FeatureVector
        };
        File.WriteAllText(metadataPath, Newtonsoft.Json.JsonConvert.SerializeObject(metadata, Newtonsoft.Json.Formatting.Indented));
    }
}

[Verb("train", HelpText = "Train the ML model")]
class TrainCommand
{
    [Value(0, Required = true, HelpText = "Path to training configuration file")]
    public string ConfigFile { get; set; } = string.Empty;

    [Option("gpu", Default = 0, HelpText = "GPU device ID")]
    public int GpuId { get; set; }

    [Option("resume", HelpText = "Resume from checkpoint")]
    public string? Checkpoint { get; set; }

    public Task<int> ExecuteAsync()
    {
        AnsiConsole.MarkupLine("[yellow]Training not yet implemented - ML models coming in Sprint 3-4[/]");
        return Task.FromResult(0);
    }
}

[Verb("evaluate", HelpText = "Evaluate a trained model")]
class EvaluateCommand
{
    [Value(0, Required = true, HelpText = "Path to trained model")]
    public string ModelPath { get; set; } = string.Empty;

    [Value(1, Required = true, HelpText = "Path to test data")]
    public string TestData { get; set; } = string.Empty;

    public Task<int> ExecuteAsync()
    {
        AnsiConsole.MarkupLine("[yellow]Evaluation not yet implemented[/]");
        return Task.FromResult(0);
    }
}

[Verb("export", HelpText = "Export model embeddings")]
class ExportCommand
{
    [Value(0, Required = true, HelpText = "Path to trained model")]
    public string ModelPath { get; set; } = string.Empty;

    [Value(1, Required = true, HelpText = "Path to data directory")]
    public string DataDir { get; set; } = string.Empty;

    public Task<int> ExecuteAsync()
    {
        AnsiConsole.MarkupLine("[yellow]Export not yet implemented[/]");
        return Task.FromResult(0);
    }
}
}
