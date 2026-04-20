using CommandLine;
using Spectre.Console;
using Serilog;
using ModL.Data.Datasets;
using ModL.Data.Pipeline;
using ModL.Core.IO;
using ModL.Data.Annotations;
using ModL.ML.Data;
using ModL.ML.Models;
using ModL.ML.Training;
using SixLabors.ImageSharp;

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
                    .Color(Spectre.Console.Color.Green));

            AnsiConsole.MarkupLine("[dim]3D Model AI Training System[/]");
            AnsiConsole.WriteLine();

            return await Parser.Default.ParseArguments<
                DownloadCommand,
                PreprocessCommand,
                CatalogCommand,
                SplitCommand,
                TrainCommand,
                EvaluateCommand,
                ExportCommand>(args)
                .MapResult(
                    (DownloadCommand cmd)    => cmd.ExecuteAsync(),
                    (PreprocessCommand cmd)  => cmd.ExecuteAsync(),
                    (CatalogCommand cmd)     => cmd.ExecuteAsync(),
                    (SplitCommand cmd)       => cmd.ExecuteAsync(),
                    (TrainCommand cmd)       => cmd.ExecuteAsync(),
                    (EvaluateCommand cmd)    => cmd.ExecuteAsync(),
                    (ExportCommand cmd)      => cmd.ExecuteAsync(),
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
                MultiViewCount  = Views,
                Normalize       = true,
                CalculateNormals = true,
                OutputDir       = OutputDir
            };

            // Find all model files for every format the factory can load
            var supported  = ModelIOFactory.SupportedLoadExtensions;
            var modelFiles = supported
                .SelectMany(ext => Directory.GetFiles(InputDir, $"*{ext}", SearchOption.AllDirectories))
                .OrderBy(f => f)
                .ToArray();

            AnsiConsole.MarkupLine($"Found [yellow]{modelFiles.Length}[/] models " +
                $"([dim]{string.Join(", ", supported)}[/])");

            var processor = new DataProcessor();
            int succeeded = 0, failed = 0;

            await AnsiConsole.Progress()
                .StartAsync(async ctx =>
                {
                    var task = ctx.AddTask("[green]Processing models[/]", maxValue: modelFiles.Length);

                    // Build input list: try sidecar .json first, then path inference
                    var inputs = modelFiles.Select(f =>
                    {
                        var jsonSidecar = Path.ChangeExtension(f, ".json");
                        var ann = File.Exists(jsonSidecar)
                            ? AnnotationParserFactory.Parse(jsonSidecar)
                            : null;   // DataProcessor.ProcessFile will call InferFromPath
                        return (f, ann);
                    });

                    var progressReporter = new Progress<BatchProgress>(p =>
                    {
                        task.Description = $"[green]Processing {p.CurrentModel}[/]";
                        task.Value        = p.Current;
                    });

                    var results = await processor.ProcessBatch(
                        inputs,
                        config,
                        progressReporter);

                    foreach (var r in results)
                    {
                        if (r.Success) succeeded++;
                        else
                        {
                            failed++;
                            Log.Warning(r.Error, "Failed to process {File}", r.FilePath);
                        }
                    }
                });

            AnsiConsole.MarkupLine(
                $"[green]✓ Processed {succeeded}/{modelFiles.Length} models[/]" +
                (failed > 0 ? $" [yellow]({failed} failed)[/]" : string.Empty));
            return 0;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Preprocessing failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return 1;
        }
    }
}

[Verb("train", HelpText = "Train the ModL model")]
class TrainCommand
{
    [Value(0, Required = true, HelpText = "Path to training config JSON (or processed dir for defaults)")]
    public string ConfigFile { get; set; } = string.Empty;

    [Option("gpu", Default = false, HelpText = "Use CUDA GPU if available")]
    public bool Gpu { get; set; }

    [Option("resume", HelpText = "Resume from this checkpoint directory")]
    public string? Checkpoint { get; set; }

    [Option("epochs", Default = 0, HelpText = "Override epoch count from config")]
    public int Epochs { get; set; }

    public async Task<int> ExecuteAsync()
    {
        try
        {
            TrainingConfig cfg;
            if (File.Exists(ConfigFile))
            {
                cfg = TrainingConfig.FromJson(ConfigFile);
            }
            else if (Directory.Exists(ConfigFile))
            {
                cfg = TrainingConfig.Default(ConfigFile);
                AnsiConsole.MarkupLine("[dim]No config file found — using defaults.[/]");
            }
            else
            {
                AnsiConsole.MarkupLine($"[red]Config file or processed dir not found: {ConfigFile}[/]");
                return 1;
            }

            if (Gpu)         cfg.Device     = "cuda";
            if (Checkpoint != null) cfg.ResumeFrom = Checkpoint;
            if (Epochs > 0)  cfg.Epochs     = Epochs;

            AnsiConsole.MarkupLine(
                $"[green]Training[/] · [dim]{cfg.Epochs} epochs · batch {cfg.BatchSize} · device {cfg.Device}[/]");

            var trainer = new Trainer(cfg);
            IReadOnlyList<EpochMetrics>? history = null;

            await AnsiConsole.Progress()
                .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(),
                         new PercentageColumn(), new ElapsedTimeColumn())
                .StartAsync(async ctx =>
                {
                    var task = ctx.AddTask("[green]Training[/]", maxValue: cfg.Epochs);

                    var progress = new Progress<TrainingProgress>(p =>
                    {
                        task.Value = p.CurrentEpoch;
                        task.Description =
                            $"[green]Epoch {p.CurrentEpoch}/{p.TotalEpochs}[/] " +
                            $"loss={p.Metrics.TrainLoss:F4} val={p.Metrics.ValTop1Accuracy * 100:F1}%";
                    });

                    history = await trainer.TrainAsync(progress);
                });

            if (history != null && history.Count > 0)
            {
                var best = history.OrderByDescending(m => m.ValTop1Accuracy).First();
                AnsiConsole.MarkupLine(
                    $"[green]✓ Training complete![/]  Best val accuracy: [yellow]{best.ValTop1Accuracy * 100:F2}%[/] (epoch {best.Epoch})");
                AnsiConsole.MarkupLine($"[dim]Checkpoints saved to: {cfg.CheckpointDir}[/]");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Training failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return 1;
        }
    }
}

[Verb("evaluate", HelpText = "Evaluate a trained ModL model on a test set")]
class EvaluateCommand
{
    [Value(0, Required = true, HelpText = "Path to checkpoint directory (contains voxel_enc.bin etc.)")]
    public string ModelPath { get; set; } = string.Empty;

    [Value(1, Required = true, HelpText = "Processed data directory")]
    public string ProcessedDir { get; set; } = string.Empty;

    [Option("config", HelpText = "Training config JSON (for model dims)")]
    public string? ConfigFile { get; set; }

    [Option("index", HelpText = "Index file (test.txt) to restrict evaluation")]
    public string? IndexFile { get; set; }

    [Option("output", HelpText = "Save evaluation JSON to this path")]
    public string? OutputJson { get; set; }

    [Option("gpu", Default = false, HelpText = "Use CUDA GPU")]
    public bool Gpu { get; set; }

    public async Task<int> ExecuteAsync()
    {
        try
        {
            var cfg = ConfigFile != null && File.Exists(ConfigFile)
                ? TrainingConfig.FromJson(ConfigFile)
                : new TrainingConfig { ProcessedDir = ProcessedDir };

            var model = new ModLModel(cfg.NumClasses, cfg.VoxelLatentDim, cfg.ViewLatentDim, cfg.EmbeddingDim);
            model.Load(ModelPath);
            var device = Gpu ? "cuda" : "cpu";
            model.ToDevice(new TorchSharp.torch.Device(Gpu ? TorchSharp.DeviceType.CUDA : TorchSharp.DeviceType.CPU));

            var testIdx   = Path.Combine(ProcessedDir, "test.txt");
            var batchCfg = new TrainingBatchConfig
            {
                ProcessedDir    = ProcessedDir,
                IndexFile       = IndexFile ?? (File.Exists(testIdx) ? testIdx : null),
                BatchSize       = cfg.BatchSize,
                VoxelResolution = cfg.VoxelResolution,
                NumViews        = cfg.NumViews,
                ViewImageSize   = cfg.ViewImageSize
            };

            var loader    = new ModelDataLoader(batchCfg);
            var evaluator = new Evaluator(model, device);

            AnsiConsole.MarkupLine("[green]Evaluating...[/]");
            var result = await evaluator.EvaluateAsync(loader);
            result.Print();

            if (OutputJson != null)
            {
                result.SaveJson(OutputJson);
                AnsiConsole.MarkupLine($"[dim]Results saved to {OutputJson}[/]");
            }

            model.Dispose();
            return 0;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Evaluation failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return 1;
        }
    }
}

[Verb("export", HelpText = "Export model embeddings to JSON")]
class ExportCommand
{
    [Value(0, Required = true, HelpText = "Path to checkpoint directory")]
    public string ModelPath { get; set; } = string.Empty;

    [Value(1, Required = true, HelpText = "Processed data directory")]
    public string DataDir { get; set; } = string.Empty;

    [Option("output", Default = "embeddings.json", HelpText = "Output JSON file path")]
    public string Output { get; set; } = "embeddings.json";

    [Option("config", HelpText = "Training config JSON")]
    public string? ConfigFile { get; set; }

    [Option("index", HelpText = "Restrict to models listed in this index file")]
    public string? IndexFile { get; set; }

    [Option("gpu", Default = false, HelpText = "Use CUDA GPU")]
    public bool Gpu { get; set; }

    public async Task<int> ExecuteAsync()
    {
        try
        {
            var cfg = ConfigFile != null && File.Exists(ConfigFile)
                ? TrainingConfig.FromJson(ConfigFile)
                : new TrainingConfig { ProcessedDir = DataDir };

            var model = new ModLModel(cfg.NumClasses, cfg.VoxelLatentDim, cfg.ViewLatentDim, cfg.EmbeddingDim);
            model.Load(ModelPath);
            model.ToDevice(new TorchSharp.torch.Device(Gpu ? TorchSharp.DeviceType.CUDA : TorchSharp.DeviceType.CPU));

            var batchCfg = new TrainingBatchConfig
            {
                ProcessedDir    = DataDir,
                IndexFile       = IndexFile,
                BatchSize       = cfg.BatchSize,
                VoxelResolution = cfg.VoxelResolution,
                NumViews        = cfg.NumViews,
                ViewImageSize   = cfg.ViewImageSize
            };

            var loader    = new ModelDataLoader(batchCfg);
            var evaluator = new Evaluator(model, Gpu ? "cuda" : "cpu");

            AnsiConsole.MarkupLine($"[green]Exporting embeddings → {Output}[/]");
            await evaluator.ExportEmbeddingsAsync(loader, Output);
            AnsiConsole.MarkupLine("[green]✓ Export complete![/]");

            model.Dispose();
            return 0;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Export failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return 1;
        }
    }
}

[Verb("catalog", HelpText = "Scan a dataset directory and show category distribution")]
class CatalogCommand
{
    [Value(0, Required = true, HelpText = "Dataset root directory")]
    public string InputDir { get; set; } = string.Empty;

    [Option("format", Default = "auto", HelpText = "Dataset format (auto, modelnet, shapenet)")]
    public string Format { get; set; } = "auto";

    public Task<int> ExecuteAsync()
    {
        try
        {
            if (!Directory.Exists(InputDir))
            {
                AnsiConsole.MarkupLine("[red]Input directory does not exist![/]");
                return Task.FromResult(1);
            }

            var supported = ModelIOFactory.SupportedLoadExtensions;
            var files = supported
                .SelectMany(ext => Directory.GetFiles(InputDir, $"*{ext}", SearchOption.AllDirectories))
                .OrderBy(f => f)
                .ToArray();

            AnsiConsole.MarkupLine($"Found [yellow]{files.Length}[/] model files in [dim]{InputDir}[/]");

            // Tally categories via path inference
            var counts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var unannotated = 0;
            foreach (var f in files)
            {
                var ann = AnnotationParserFactory.InferFromPath(f);
                if (ann != null)
                {
                    var cat = ann.Category ?? "unknown";
                    counts[cat] = counts.GetValueOrDefault(cat) + 1;
                }
                else
                {
                    unannotated++;
                }
            }

            if (counts.Count == 0)
            {
                AnsiConsole.MarkupLine("[yellow]No category information could be inferred. Try specifying --format.[/]");
                return Task.FromResult(0);
            }

            var table = new Spectre.Console.Table()
                .AddColumn("Category")
                .AddColumn(new Spectre.Console.TableColumn("Count").RightAligned())
                .AddColumn(new Spectre.Console.TableColumn("%").RightAligned());

            int total = counts.Values.Sum();
            foreach (var (cat, count) in counts.OrderByDescending(x => x.Value))
                table.AddRow(cat, count.ToString(), $"{100.0 * count / total:F1}%");

            if (unannotated > 0)
                table.AddRow("[dim]unannotated[/]", unannotated.ToString(), $"{100.0 * unannotated / files.Length:F1}%");

            AnsiConsole.Write(table);
            AnsiConsole.MarkupLine($"[dim]Total annotated: {total}  |  Categories: {counts.Count}[/]");
            return Task.FromResult(0);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Catalog failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return Task.FromResult(1);
        }
    }
}

[Verb("split", HelpText = "Split a processed output directory into train/val/test index files")]
class SplitCommand
{
    [Value(0, Required = true, HelpText = "Processed output directory (from preprocess command)")]
    public string ProcessedDir { get; set; } = string.Empty;

    [Option("train", Default = 0.8f, HelpText = "Training fraction (default 0.8)")]
    public float Train { get; set; }

    [Option("val", Default = 0.1f, HelpText = "Validation fraction (default 0.1)")]
    public float Val { get; set; }

    [Option("test", Default = 0.1f, HelpText = "Test fraction (default 0.1)")]
    public float Test { get; set; }

    [Option("seed", Default = 42, HelpText = "Random seed for reproducibility")]
    public int Seed { get; set; }

    public Task<int> ExecuteAsync()
    {
        try
        {
            if (!Directory.Exists(ProcessedDir))
            {
                AnsiConsole.MarkupLine("[red]Processed directory does not exist![/]");
                return Task.FromResult(1);
            }

            var store   = new ProcessedModelStore();
            var entries = store.LoadAll(ProcessedDir, loadViews: false).ToList();

            if (entries.Count == 0)
            {
                AnsiConsole.MarkupLine("[yellow]No processed models found in directory.[/]");
                return Task.FromResult(0);
            }

            var splitter = new DatasetSplitter(Train, Val, Test, Seed);
            var split    = splitter.Split(entries, e => e.Annotation?.Category ?? "unknown");

            File.WriteAllLines(Path.Combine(ProcessedDir, "train.txt"), split.Train.Select(e => e.ModelId));
            File.WriteAllLines(Path.Combine(ProcessedDir, "val.txt"),   split.Val.Select(e => e.ModelId));
            File.WriteAllLines(Path.Combine(ProcessedDir, "test.txt"),  split.Test.Select(e => e.ModelId));

            AnsiConsole.MarkupLine("[green]✓ Split complete![/]");
            split.PrintSummary();
            AnsiConsole.MarkupLine($"[dim]Written: train.txt / val.txt / test.txt → {ProcessedDir}[/]");
            return Task.FromResult(0);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Split failed");
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
            return Task.FromResult(1);
        }
    }
}
