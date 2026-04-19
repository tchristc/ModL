namespace ModL.Data.Pipeline;

/// <summary>
/// Splits a collection of model records into train / validation / test sets.
/// Sampling is stratified by category so every split contains a proportional
/// share of each class. The split is deterministic given the same seed.
/// </summary>
public class DatasetSplitter
{
    public float TrainFraction { get; }
    public float ValFraction   { get; }
    public float TestFraction  { get; }
    public int   Seed          { get; }

    public DatasetSplitter(
        float trainFraction = 0.8f,
        float valFraction   = 0.1f,
        float testFraction  = 0.1f,
        int   seed          = 42)
    {
        if (MathF.Abs(trainFraction + valFraction + testFraction - 1f) > 0.001f)
            throw new ArgumentException("Train + val + test fractions must sum to 1.0");

        TrainFraction = trainFraction;
        ValFraction   = valFraction;
        TestFraction  = testFraction;
        Seed          = seed;
    }

    /// <summary>
    /// Splits a list of entries into three named buckets.
    /// <paramref name="getCategory"/> extracts the class label used for stratification.
    /// </summary>
    public DatasetSplit<T> Split<T>(IEnumerable<T> items, Func<T, string> getCategory)
    {
        var rng = new Random(Seed);

        // Group by category for stratified sampling
        var groups = items
            .GroupBy(getCategory)
            .ToDictionary(g => g.Key, g => g.OrderBy(_ => rng.Next()).ToList());

        var train = new List<T>();
        var val   = new List<T>();
        var test  = new List<T>();

        foreach (var (_, group) in groups)
        {
            int n        = group.Count;
            int trainEnd = (int)MathF.Round(n * TrainFraction);
            int valEnd   = trainEnd + (int)MathF.Round(n * ValFraction);

            train.AddRange(group[..trainEnd]);
            val.AddRange(  group[trainEnd..valEnd]);
            test.AddRange( group[valEnd..]);
        }

        // Shuffle each split independently so categories are interleaved
        Shuffle(train, rng);
        Shuffle(val,   rng);
        Shuffle(test,  rng);

        return new DatasetSplit<T>(train, val, test);
    }

    private static void Shuffle<T>(List<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}

/// <summary>Immutable result of a dataset split.</summary>
public sealed class DatasetSplit<T>
{
    public IReadOnlyList<T> Train { get; }
    public IReadOnlyList<T> Val   { get; }
    public IReadOnlyList<T> Test  { get; }

    public int Total => Train.Count + Val.Count + Test.Count;

    public DatasetSplit(IReadOnlyList<T> train, IReadOnlyList<T> val, IReadOnlyList<T> test)
    {
        Train = train;
        Val   = val;
        Test  = test;
    }

    public void PrintSummary()
    {
        Console.WriteLine($"  Train : {Train.Count,6} ({100f * Train.Count / Total:F1}%)");
        Console.WriteLine($"  Val   : {Val.Count,6}   ({100f * Val.Count   / Total:F1}%)");
        Console.WriteLine($"  Test  : {Test.Count,6}  ({100f * Test.Count  / Total:F1}%)");
        Console.WriteLine($"  Total : {Total,6}");
    }
}
