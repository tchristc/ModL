using ModL.ML.Data;
using ModL.ML.Models;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ModL.ML.Training;

/// <summary>
/// Drives the full training loop for <see cref="ModLModel"/>.
///
/// Responsibilities:
///   • Build or restore model + optimiser state
///   • Run train / validation epochs
///   • Report per-epoch metrics via IProgress
///   • Save best checkpoint and periodic checkpoints
///   • Honour CancellationToken for graceful stop
/// </summary>
public sealed class Trainer
{
    private readonly TrainingConfig _cfg;

    public Trainer(TrainingConfig cfg) => _cfg = cfg;

    // -----------------------------------------------------------------------
    // Public entry point
    // -----------------------------------------------------------------------

    public async Task<IReadOnlyList<EpochMetrics>> TrainAsync(
        IProgress<TrainingProgress>? progress   = null,
        CancellationToken            ct          = default)
    {
        // ── Resolve device ────────────────────────────────────────────────
        var device = ResolveDevice(_cfg.Device);

        // ── Data loaders ──────────────────────────────────────────────────
        var trainCfg = MakeBatchConfig(_cfg.TrainIndexFile);
        var valCfg   = MakeBatchConfig(_cfg.ValIndexFile);

        var trainLoader = new ModelDataLoader(trainCfg);
        var valLoader   = new ModelDataLoader(valCfg, trainLoader.LabelMap);

        // ── Model ─────────────────────────────────────────────────────────
        var model = new ModLModel(
            trainLoader.LabelMap.Count > 0 ? trainLoader.LabelMap.Count : _cfg.NumClasses,
            _cfg.VoxelLatentDim,
            _cfg.ViewLatentDim,
            _cfg.EmbeddingDim);
        model.ToDevice(device);

        int startEpoch = 0;
        if (_cfg.ResumeFrom != null && Directory.Exists(_cfg.ResumeFrom))
        {
            model.Load(_cfg.ResumeFrom);
            startEpoch = ReadEpochMarker(_cfg.ResumeFrom);
        }

        // ── Optimiser & scheduler ─────────────────────────────────────────
        var parameters = model.VoxelEnc.parameters()
            .Concat(model.ViewEnc.parameters())
            .Concat(model.Fusion.parameters());

        var optimiser = optim.Adam(parameters,
            lr: _cfg.LearningRate, weight_decay: _cfg.WeightDecay);

        var scheduler = BuildScheduler(optimiser);

        // ── Loss ──────────────────────────────────────────────────────────
        var lossFunc = CrossEntropyLoss();

        // ── Metrics history ───────────────────────────────────────────────
        var history = new List<EpochMetrics>();
        double bestValAcc = 0;

        Directory.CreateDirectory(_cfg.CheckpointDir);

        // Save config alongside checkpoints for reproducibility
        var configPath = Path.Combine(_cfg.CheckpointDir, "config.json");
        if (!File.Exists(configPath))
            _cfg.SaveJson(configPath);

        // ── Epoch loop ────────────────────────────────────────────────────
        for (int epoch = startEpoch; epoch < _cfg.Epochs; epoch++)
        {
            ct.ThrowIfCancellationRequested();

            var sw     = System.Diagnostics.Stopwatch.StartNew();
            var trainL = await RunEpochAsync(model, trainLoader, optimiser, lossFunc, device,
                                             isTraining: true, ct);
            var (valL, top1, top5) = await RunEvalAsync(model, valLoader, lossFunc, device, ct);
            sw.Stop();

            scheduler?.step();

            var metrics = new EpochMetrics(epoch + 1, trainL, valL, top1, top5, sw.Elapsed);
            history.Add(metrics);

            progress?.Report(new TrainingProgress(metrics, epoch + 1, _cfg.Epochs));

            // Save periodic checkpoint
            if (_cfg.SaveEveryEpochs > 0 && (epoch + 1) % _cfg.SaveEveryEpochs == 0)
            {
                var ckpt = Path.Combine(_cfg.CheckpointDir, $"epoch_{epoch + 1:D4}");
                model.Save(ckpt);
                WriteEpochMarker(ckpt, epoch + 1);
            }

            // Save best model
            if (top1 > bestValAcc)
            {
                bestValAcc = top1;
                var best = Path.Combine(_cfg.CheckpointDir, "best");
                model.Save(best);
                WriteEpochMarker(best, epoch + 1);
            }
        }

        // Final save
        var final = Path.Combine(_cfg.CheckpointDir, "final");
        model.Save(final);

        model.Dispose();
        return history;
    }

    // -----------------------------------------------------------------------
    // Epoch helpers
    // -----------------------------------------------------------------------

    private static async Task<double> RunEpochAsync(
        ModLModel model,
        ModelDataLoader loader,
        optim.Optimizer optimiser,
        TorchSharp.Modules.CrossEntropyLoss loss,
        Device device,
        bool isTraining,
        CancellationToken ct)
    {
        return await Task.Run(() =>
        {
            if (isTraining) model.Train(); else model.Eval();

            double totalLoss = 0;
            int    batches   = 0;

            foreach (var batch in loader.GetBatches(shuffle: isTraining))
            {
                ct.ThrowIfCancellationRequested();

                using (batch)
                using (var voxels = batch.Voxels.to(device))
                using (var views  = batch.Views.to(device))
                using (var labels = batch.Labels.to(device))
                {
                    if (isTraining)
                    {
                        optimiser.zero_grad();
                        using var logits = model.Forward(voxels, views);
                        using var l      = loss.forward(logits, labels);
                        l.backward();
                        optimiser.step();
                        totalLoss += l.item<float>();
                    }
                    else
                    {
                        using var noGrad = no_grad();
                        using var logits = model.Forward(voxels, views);
                        using var l      = loss.forward(logits, labels);
                        totalLoss += l.item<float>();
                    }
                }

                batches++;
            }

            return batches > 0 ? totalLoss / batches : 0;
        }, ct);
    }

    private static async Task<(double loss, double top1, double top5)> RunEvalAsync(
        ModLModel model,
        ModelDataLoader loader,
        TorchSharp.Modules.CrossEntropyLoss loss,
        Device device,
        CancellationToken ct)
    {
        return await Task.Run(() =>
        {
            model.Eval();
            double totalLoss = 0;
            int total = 0, correct1 = 0, correct5 = 0, batches = 0;

            foreach (var batch in loader.GetBatches(shuffle: false))
            {
                ct.ThrowIfCancellationRequested();

                using (batch)
                using (var voxels = batch.Voxels.to(device))
                using (var views  = batch.Views.to(device))
                using (var labels = batch.Labels.to(device))
                using (var noGrad = no_grad())
                {
                    using var logits = model.Forward(voxels, views);
                    using var l      = loss.forward(logits, labels);
                    totalLoss += l.item<float>();

                    // Top-1
                    using var pred1 = logits.argmax(1);
                    correct1 += (int)pred1.eq(labels).sum().item<long>();

                    // Top-5
                    int k5 = (int)Math.Min(5L, logits.shape[1]);
                    using var top5Indices = logits.topk(k5, 1).indexes;
                    for (int i = 0; i < labels.shape[0]; i++)
                    {
                        long lbl = labels[i].item<long>();
                        for (int k = 0; k < top5Indices.shape[1]; k++)
                        {
                            if (top5Indices[i, k].item<long>() == lbl) { correct5++; break; }
                        }
                    }

                    total  += (int)labels.shape[0];
                }

                batches++;
            }

            double avgLoss = batches > 0 ? totalLoss / batches : 0;
            double t1      = total > 0 ? (double)correct1 / total : 0;
            double t5      = total > 0 ? (double)correct5 / total : 0;
            return (avgLoss, t1, t5);
        }, ct);
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    private TrainingBatchConfig MakeBatchConfig(string? indexFile) => new()
    {
        ProcessedDir    = _cfg.ProcessedDir,
        IndexFile       = indexFile,
        BatchSize       = _cfg.BatchSize,
        VoxelResolution = _cfg.VoxelResolution,
        NumViews        = _cfg.NumViews,
        ViewImageSize   = _cfg.ViewImageSize
    };

    private optim.lr_scheduler.LRScheduler? BuildScheduler(optim.Optimizer opt) =>
        _cfg.LrSchedule.ToLowerInvariant() switch
        {
            "cosine" => optim.lr_scheduler.CosineAnnealingLR(opt, _cfg.Epochs),
            "step"   => optim.lr_scheduler.StepLR(opt, _cfg.LrStepEvery, _cfg.LrStepGamma),
            _        => null
        };

    private static Device ResolveDevice(string name)
    {
        var lower = name.ToLowerInvariant();
        if (lower == "cpu")
            return new Device(TorchSharp.DeviceType.CPU);
        if (lower is "cuda" or "gpu")
            return ResolveCudaOrFallback();
        if (lower.StartsWith("cuda:"))
            return ResolveCudaOrFallback(int.Parse(lower["cuda:".Length..]));
        return new Device(TorchSharp.DeviceType.CPU);
    }

    private static Device ResolveCudaOrFallback(int index = -1)
    {
        if (cuda.is_available())
            return index >= 0
                ? new Device(TorchSharp.DeviceType.CUDA, index)
                : new Device(TorchSharp.DeviceType.CUDA);
        Console.WriteLine("[WARNING] CUDA is not available — falling back to CPU.");
        return new Device(TorchSharp.DeviceType.CPU);
    }

    private static void WriteEpochMarker(string dir, int epoch)
        => File.WriteAllText(Path.Combine(dir, "epoch.txt"), epoch.ToString());

    private static int ReadEpochMarker(string dir)
    {
        var f = Path.Combine(dir, "epoch.txt");
        return File.Exists(f) && int.TryParse(File.ReadAllText(f).Trim(), out var e) ? e : 0;
    }
}

/// <summary>Live progress event fired after each epoch.</summary>
public sealed record TrainingProgress(
    EpochMetrics Metrics,
    int          CurrentEpoch,
    int          TotalEpochs)
{
    public double PercentComplete => TotalEpochs > 0 ? 100.0 * CurrentEpoch / TotalEpochs : 0;
}
