using System.Security.Cryptography;
using System.Text.Json;
using IngestionApi.Models;

namespace IngestionApi.Services;

/// <summary>
/// Filesystem-backed Bronze store. Mirrors the Python <c>DocumentStore</c>: each
/// upload is assigned a UUID, written verbatim under the Bronze directory, and
/// accompanied by a <c>manifest.json</c> with the content hash and size.
/// </summary>
public sealed class FileDocumentStore
{
    private const string ManifestName = "manifest.json";
    // The PDF file signature ("%PDF-"); uploads must start with it.
    private static readonly byte[] PdfSignature = "%PDF-"u8.ToArray();
    private static readonly JsonSerializerOptions JsonOptions =
        new(JsonSerializerDefaults.Web) { WriteIndented = true };

    private readonly string _bronzeRoot;

    /// <summary>Resolve the Bronze directory from configuration or the shared env var.</summary>
    public FileDocumentStore(IConfiguration configuration)
    {
        var dataDir =
            Environment.GetEnvironmentVariable("PROPINTELLI_DATA_DIR")
            ?? configuration["Storage:DataDir"]
            ?? "./data";
        _bronzeRoot = Path.Combine(dataDir, "bronze");
        Directory.CreateDirectory(_bronzeRoot);
    }

    /// <summary>Store raw bytes as a new Bronze document and write its manifest.</summary>
    /// <exception cref="ArgumentException">If the payload is empty.</exception>
    public async Task<BronzeDocument> IngestAsync(
        Stream content,
        string fileName,
        CancellationToken cancellationToken)
    {
        using var buffer = new MemoryStream();
        await content.CopyToAsync(buffer, cancellationToken);
        var bytes = buffer.ToArray();
        if (bytes.Length == 0)
        {
            throw new ArgumentException("Refusing to ingest an empty document.", nameof(content));
        }

        if (!HasPdfSignature(bytes))
        {
            throw new ArgumentException(
                "Only PDF documents are accepted (missing %PDF- signature).", nameof(content));
        }

        var documentId = Guid.NewGuid().ToString("N");
        var directory = Path.Combine(_bronzeRoot, documentId);
        Directory.CreateDirectory(directory);

        var suffix = Path.GetExtension(fileName);
        if (string.IsNullOrEmpty(suffix))
        {
            suffix = ".bin";
        }

        var storedPath = Path.Combine(directory, $"original{suffix}");
        await File.WriteAllBytesAsync(storedPath, bytes, cancellationToken);

        var document = new BronzeDocument(
            DocumentId: documentId,
            SourceDocument: Path.GetFileName(fileName),
            StoredPath: storedPath,
            Sha256: Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(),
            SizeBytes: bytes.Length,
            ReceivedAt: DateTimeOffset.UtcNow);

        await File.WriteAllTextAsync(
            Path.Combine(directory, ManifestName),
            JsonSerializer.Serialize(document, JsonOptions),
            cancellationToken);

        return document;
    }

    /// <summary>Read a stored document's manifest, or <c>null</c> if it does not exist.</summary>
    public async Task<BronzeDocument?> GetAsync(string documentId, CancellationToken cancellationToken)
    {
        var manifestPath = Path.Combine(_bronzeRoot, documentId, ManifestName);
        if (!File.Exists(manifestPath))
        {
            return null;
        }

        await using var stream = File.OpenRead(manifestPath);
        return await JsonSerializer.DeserializeAsync<BronzeDocument>(
            stream, JsonOptions, cancellationToken);
    }

    private static bool HasPdfSignature(ReadOnlySpan<byte> bytes) =>
        bytes.Length >= PdfSignature.Length && bytes[..PdfSignature.Length].SequenceEqual(PdfSignature);
}
