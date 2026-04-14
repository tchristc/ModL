using Serilog;

namespace ModL.Data.Datasets;

/// <summary>
/// Downloads ShapeNet dataset
/// </summary>
public class ShapeNetDownloader : IDatasetDownloader
{
    private readonly ILogger _logger;

    public string DatasetName => "ShapeNet";

    public ShapeNetDownloader(ILogger? logger = null)
    {
        _logger = logger ?? Log.Logger;
    }

    public async Task DownloadAsync(DatasetConfig config, IProgress<DownloadProgress>? progress = null)
    {
        _logger.Information("Starting ShapeNet download to {Path}", config.LocalPath);

        // Create output directory
        Directory.CreateDirectory(config.LocalPath);

        progress?.Report(new DownloadProgress
        {
            Message = "ShapeNet requires manual download from https://shapenet.org/"
        });

        // ShapeNet requires registration and manual download
        // Provide instructions to user
        var readmePath = Path.Combine(config.LocalPath, "README.txt");
        await File.WriteAllTextAsync(readmePath, @"
ShapeNet Dataset Download Instructions
======================================

ShapeNet requires registration and cannot be downloaded automatically.

1. Visit https://shapenet.org/
2. Create an account or sign in
3. Navigate to the downloads section
4. Download ShapeNetCore.v2 or ShapeNetSem
5. Extract the downloaded files to this directory

Expected directory structure:
{LocalPath}/
    ├── 02691156/  (airplane category)
    ├── 02958343/  (car category)
    ├── ...
    └── taxonomy.json

Once downloaded, run the preprocessing command to prepare the data for training.
");

        _logger.Information("ShapeNet download instructions written to {Path}", readmePath);
        _logger.Warning("ShapeNet requires manual download. See README.txt for instructions.");
    }

    public bool IsDownloaded(DatasetConfig config)
    {
        if (!Directory.Exists(config.LocalPath))
            return false;

        // Check for taxonomy.json or any category directories
        var taxonomyPath = Path.Combine(config.LocalPath, "taxonomy.json");
        if (File.Exists(taxonomyPath))
            return true;

        // Check for category directories (ShapeNet format)
        var dirs = Directory.GetDirectories(config.LocalPath);
        return dirs.Any(d => Path.GetFileName(d).Length == 8 && Path.GetFileName(d).All(char.IsDigit));
    }
}

/// <summary>
/// Downloads ModelNet dataset
/// </summary>
public class ModelNetDownloader : IDatasetDownloader
{
    private readonly ILogger _logger;
    private readonly HttpClient _httpClient;

    public string DatasetName => "ModelNet";

    public ModelNetDownloader(HttpClient? httpClient = null, ILogger? logger = null)
    {
        _httpClient = httpClient ?? new HttpClient();
        _logger = logger ?? Log.Logger;
    }

    public async Task DownloadAsync(DatasetConfig config, IProgress<DownloadProgress>? progress = null)
    {
        _logger.Information("Starting ModelNet download to {Path}", config.LocalPath);

        // Create output directory
        Directory.CreateDirectory(config.LocalPath);

        // ModelNet40 download URL
        var modelNet40Url = "http://modelnet.cs.princeton.edu/ModelNet40.zip";
        var zipPath = Path.Combine(config.LocalPath, "ModelNet40.zip");

        try
        {
            // Download the zip file
            progress?.Report(new DownloadProgress { Message = "Downloading ModelNet40.zip..." });

            using var response = await _httpClient.GetAsync(modelNet40Url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? 0;
            var bytesDownloaded = 0L;

            using var stream = await response.Content.ReadAsStreamAsync();
            using var fileStream = new FileStream(zipPath, FileMode.Create, FileAccess.Write, FileShare.None);

            var buffer = new byte[8192];
            int bytesRead;

            while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0)
            {
                await fileStream.WriteAsync(buffer, 0, bytesRead);
                bytesDownloaded += bytesRead;

                progress?.Report(new DownloadProgress
                {
                    CurrentFile = "ModelNet40.zip",
                    BytesDownloaded = bytesDownloaded,
                    TotalBytes = totalBytes,
                    Message = $"Downloading: {bytesDownloaded / (1024 * 1024)}MB / {totalBytes / (1024 * 1024)}MB"
                });
            }

            _logger.Information("Downloaded ModelNet40.zip");

            // Extract the zip file
            progress?.Report(new DownloadProgress { Message = "Extracting..." });
            System.IO.Compression.ZipFile.ExtractToDirectory(zipPath, config.LocalPath);

            // Delete the zip file
            File.Delete(zipPath);

            _logger.Information("ModelNet40 dataset downloaded and extracted successfully");
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Failed to download ModelNet dataset");
            throw;
        }
    }

    public bool IsDownloaded(DatasetConfig config)
    {
        if (!Directory.Exists(config.LocalPath))
            return false;

        // Check for ModelNet40 directory structure
        var modelNet40Path = Path.Combine(config.LocalPath, "ModelNet40");
        return Directory.Exists(modelNet40Path);
    }
}

/// <summary>
/// Factory for creating dataset downloaders
/// </summary>
public static class DatasetDownloaderFactory
{
    private static readonly Dictionary<string, Func<IDatasetDownloader>> _downloaders = new()
    {
        ["shapenet"] = () => new ShapeNetDownloader(),
        ["modelnet"] = () => new ModelNetDownloader()
    };

    public static IDatasetDownloader Create(string datasetName)
    {
        var key = datasetName.ToLowerInvariant();
        if (_downloaders.TryGetValue(key, out var factory))
        {
            return factory();
        }

        throw new NotSupportedException($"Dataset '{datasetName}' is not supported");
    }

    public static string[] SupportedDatasets => _downloaders.Keys.ToArray();
}
