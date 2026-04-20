using ModL.ML.Data;
using ModL.ML.Models;
using Newtonsoft.Json;
using static TorchSharp.torch;

namespace ModL.ML.Training;

/// <summary>
/// Evaluates a trained <see cref="ModLModel"/> and can export embeddings.
///
/// Produces:
///   • Overall top-1 / top-5 accuracy
///   • Per-class precision, recall, F1
///   • Confusion matrix (as int[numClasses, numClasses])
///   • Embedding export (modelId → float[EmbeddingDim]) as JSON
/// </summary>
public sealed class Evaluator
{
    private readonly ModLModel _model;
    private readonly Device    _device;

    public Evaluator(ModLModel model, string device = "cpu")
    {
        _model  = model;
        var lower = device.ToLowerInvariant();
        _device = lower is "cuda" or "gpu"
            ? new Device(TorchSharp.DeviceType.CUDA)
            : lower.StartsWith("cuda:")
                ? new Device(TorchSharp.DeviceType.CUDA, int.Parse(lower["cuda:".Length..]))
                : new Device(TorchSharp.DeviceType.CPU);
    }

    // -----------------------------------------------------------------------
    // Classification metrics
    // -----------------------------------------------------------------------

    /// <summary>
    /// Runs inference over all batches from <paramref name="loader"/> and
    /// returns full evaluation results.
    /// </summary>
    public async Task<EvaluationResult> EvaluateAsync(
        ModelDataLoader loader,
        CancellationToken ct = default)
    {
        return await Task.Run(() =>
        {
            _model.Eval();

            var labelMap    = loader.LabelMap;
            int numClasses  = Math.Max(labelMap.Count, 1);
            var confusion   = new int[numClasses, numClasses];

            int total = 0, correct1 = 0, correct5 = 0;

            foreach (var batch in loader.GetBatches(shuffle: false))
            {
                ct.ThrowIfCancellationRequested();

                using (batch)
                using (var voxels = batch.Voxels.to(_device))
                using (var views  = batch.Views.to(_device))
                using (var labels = batch.Labels.to(_device))
                using (var noGrad = no_grad())
                {
                    using var logits   = _model.Forward(voxels, views);
                    using var pred1    = logits.argmax(1);
                    using var pred1Cpu = pred1.cpu();
                    using var lblCpu   = labels.cpu();

                    int batchSize = (int)labels.shape[0];

                    for (int i = 0; i < batchSize; i++)
                    {
                        int predicted = (int)pred1Cpu[i].item<long>();
                        int actual    = (int)lblCpu[i].item<long>();

                        if (predicted < numClasses && actual < numClasses)
                            confusion[actual, predicted]++;
                    }

                    correct1 += (int)pred1.eq(labels).sum().item<long>();

                    int k5 = (int)Math.Min(5L, (long)numClasses);
                    using var topKIdx = logits.topk(k5, 1).indexes;
                    for (int i = 0; i < batchSize; i++)
                    {
                        long lbl = lblCpu[i].item<long>();
                        for (int k = 0; k < topKIdx.shape[1]; k++)
                        {
                            if (topKIdx[i, k].item<long>() == lbl) { correct5++; break; }
                        }
                    }

                    total += batchSize;
                }
            }

            double top1 = total > 0 ? (double)correct1 / total : 0;
            double top5 = total > 0 ? (double)correct5 / total : 0;

            var perClass = ComputePerClassMetrics(confusion, labelMap, numClasses);

            return new EvaluationResult(top1, top5, confusion, perClass, total);
        }, ct);
    }

    // -----------------------------------------------------------------------
    // Embedding export
    // -----------------------------------------------------------------------

    /// <summary>
    /// Runs all models in <paramref name="loader"/> through the fusion encoder
    /// (no classifier head) and writes a JSON file mapping modelId → float[].
    /// </summary>
    public async Task ExportEmbeddingsAsync(
        ModelDataLoader loader,
        string          outputJsonPath,
        CancellationToken ct = default)
    {
        var embeddings = new Dictionary<string, float[]>();

        await Task.Run(() =>
        {
            _model.Eval();

            foreach (var batch in loader.GetBatches(shuffle: false))
            {
                ct.ThrowIfCancellationRequested();

                // We need model IDs — iterate single items for traceability
                // (batches don't carry string ids in the tensor, so we store index order)
                using (batch)
                using (var voxels = batch.Voxels.to(_device))
                using (var views  = batch.Views.to(_device))
                using (var noGrad = no_grad())
                using (var emb    = _model.Embed(voxels, views))
                using (var embCpu = emb.cpu())
                {
                    int b = (int)embCpu.shape[0];
                    int d = (int)embCpu.shape[1];
                    for (int i = 0; i < b; i++)
                    {
                        var vec = new float[d];
                        for (int j = 0; j < d; j++)
                            vec[j] = embCpu[i, j].item<float>();

                        embeddings[$"item_{embeddings.Count:D6}"] = vec;
                    }
                }
            }
        }, ct);

        Directory.CreateDirectory(Path.GetDirectoryName(outputJsonPath) ?? ".");
        File.WriteAllText(outputJsonPath,
            JsonConvert.SerializeObject(embeddings, Formatting.Indented));
    }

    // -----------------------------------------------------------------------
    // Per-class metrics
    // -----------------------------------------------------------------------

    private static IReadOnlyList<ClassMetrics> ComputePerClassMetrics(
        int[,] confusion,
        IReadOnlyDictionary<string, int> labelMap,
        int numClasses)
    {
        var reverseMap = labelMap.ToDictionary(kv => kv.Value, kv => kv.Key);
        var result     = new List<ClassMetrics>(numClasses);

        for (int c = 0; c < numClasses; c++)
        {
            int tp = confusion[c, c];
            int fp = 0, fn = 0;
            for (int j = 0; j < numClasses; j++) if (j != c) fp += confusion[j, c];
            for (int j = 0; j < numClasses; j++) if (j != c) fn += confusion[c, j];

            double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
            double recall    = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
            double f1        = precision + recall > 0
                ? 2 * precision * recall / (precision + recall) : 0;

            string name = reverseMap.TryGetValue(c, out var n) ? n : $"class_{c}";
            result.Add(new ClassMetrics(name, tp + fn, precision, recall, f1));
        }

        return result;
    }
}

// -----------------------------------------------------------------------
// Result types
// -----------------------------------------------------------------------

/// <summary>Full evaluation results from one pass over a test set.</summary>
public sealed class EvaluationResult
{
    public double Top1Accuracy    { get; }
    public double Top5Accuracy    { get; }
    public int[,] ConfusionMatrix { get; }
    public IReadOnlyList<ClassMetrics> PerClass { get; }
    public int TotalSamples       { get; }

    public EvaluationResult(
        double top1, double top5,
        int[,] confusion,
        IReadOnlyList<ClassMetrics> perClass,
        int total)
    {
        Top1Accuracy   = top1;
        Top5Accuracy   = top5;
        ConfusionMatrix = confusion;
        PerClass       = perClass;
        TotalSamples   = total;
    }

    public void Print()
    {
        Console.WriteLine($"  Samples : {TotalSamples}");
        Console.WriteLine($"  Top-1   : {Top1Accuracy * 100:F2}%");
        Console.WriteLine($"  Top-5   : {Top5Accuracy * 100:F2}%");
        Console.WriteLine();
        Console.WriteLine($"  {"Category",-20} {"Prec",8} {"Recall",8} {"F1",8} {"Support",8}");
        Console.WriteLine(new string('-', 60));
        foreach (var c in PerClass.OrderByDescending(x => x.Support))
            Console.WriteLine($"  {c.ClassName,-20} {c.Precision,8:F3} {c.Recall,8:F3} {c.F1,8:F3} {c.Support,8}");
    }

    public void SaveJson(string path)
    {
        var dto = new
        {
            Top1Accuracy,
            Top5Accuracy,
            TotalSamples,
            PerClass
        };
        File.WriteAllText(path, JsonConvert.SerializeObject(dto, Formatting.Indented));
    }
}

/// <summary>Per-category classification metrics.</summary>
public sealed record ClassMetrics(
    string ClassName,
    int    Support,
    double Precision,
    double Recall,
    double F1);
