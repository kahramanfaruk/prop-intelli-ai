namespace IngestionApi.Models;

/// <summary>Metadata describing a document stored in the Bronze layer.</summary>
/// <remarks>Serialised as the per-document <c>manifest.json</c>; mirrors the Python
/// <c>BronzeDocument</c> so both services share one Bronze contract.</remarks>
public sealed record BronzeDocument(
    string DocumentId,
    string SourceDocument,
    string StoredPath,
    string Sha256,
    long SizeBytes,
    DateTimeOffset ReceivedAt);

/// <summary>Response returned after a successful upload.</summary>
public sealed record UploadResponse(string DocumentId, string Status, string Sha256, long SizeBytes);

/// <summary>Status of a previously ingested document.</summary>
public sealed record DocumentStatus(string DocumentId, string Status, DateTimeOffset ReceivedAt);

/// <summary>A property payload submitted to the validation endpoint.</summary>
/// <remarks>JSON keys are snake_case (configured globally) to match the Python schema.</remarks>
public sealed record ValidatePropertyRequest
{
    public decimal? PriceEur { get; init; }
    public double? LivingAreaSqm { get; init; }
    public double? Rooms { get; init; }
    public int? YearBuilt { get; init; }
    public string? PostalCode { get; init; }
    public string? City { get; init; }
    public string? ListingType { get; init; }
}

/// <summary>A single validation finding.</summary>
public sealed record ValidationFinding(string RuleId, string? Field, string Severity, string Message);

/// <summary>The result of validating a property payload.</summary>
public sealed record ValidationResponse(bool IsValid, IReadOnlyList<ValidationFinding> Findings);
