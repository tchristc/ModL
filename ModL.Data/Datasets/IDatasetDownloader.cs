namespace ModL.Data.Datasets;

/// <summary>
/// Configuration for a 3D model dataset
/// </summary>
public class DatasetConfig
{
    public string Name { get; set; } = string.Empty;
    public string SourceUrl { get; set; } = string.Empty;
    public string LocalPath { get; set; } = string.Empty;
    public string[] Categories { get; set; } = Array.Empty<string>();
    public string Format { get; set; } = "obj";
    public bool RequiresAuth { get; set; } = false;
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// Interface for downloading 3D model datasets
/// </summary>
public interface IDatasetDownloader
{
    /// <summary>
    /// Downloads a dataset asynchronously
    /// </summary>
    Task DownloadAsync(DatasetConfig config, IProgress<DownloadProgress>? progress = null);

    /// <summary>
    /// Checks if the dataset is already downloaded
    /// </summary>
    bool IsDownloaded(DatasetConfig config);

    /// <summary>
    /// Gets the name of the dataset source this downloader handles
    /// </summary>
    string DatasetName { get; }
}

/// <summary>
/// Progress information for dataset downloads
/// </summary>
public class DownloadProgress
{
    public string CurrentFile { get; set; } = string.Empty;
    public long BytesDownloaded { get; set; }
    public long TotalBytes { get; set; }
    public double PercentComplete => TotalBytes > 0 ? (double)BytesDownloaded / TotalBytes * 100 : 0;
    public string Message { get; set; } = string.Empty;
}
