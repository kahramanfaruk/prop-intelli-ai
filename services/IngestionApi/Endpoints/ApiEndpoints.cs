using IngestionApi.Models;
using IngestionApi.Services;

namespace IngestionApi.Endpoints;

/// <summary>Maps the HTTP endpoints of the ingestion microservice.</summary>
public static class ApiEndpoints
{
    /// <summary>Register all routes on the application.</summary>
    public static void MapApiEndpoints(this WebApplication app)
    {
        app.MapGet("/health", () => Results.Ok(new { status = "healthy" }))
            .WithName("Health")
            .WithTags("System");

        app.MapPost(
                "/api/documents/upload",
                async (IFormFile file, FileDocumentStore store, CancellationToken ct) =>
                {
                    if (file.Length == 0)
                    {
                        return Results.BadRequest(new { error = "A non-empty file is required." });
                    }

                    await using var stream = file.OpenReadStream();
                    try
                    {
                        var document = await store.IngestAsync(stream, file.FileName, ct);
                        return Results.Ok(new UploadResponse(
                            document.DocumentId, "received", document.Sha256, document.SizeBytes));
                    }
                    catch (ArgumentException ex)
                    {
                        return Results.BadRequest(new { error = ex.Message });
                    }
                })
            .WithName("UploadDocument")
            .WithTags("Documents")
            .DisableAntiforgery();

        app.MapGet(
                "/api/documents/{documentId}/status",
                async (string documentId, FileDocumentStore store, CancellationToken ct) =>
                {
                    var document = await store.GetAsync(documentId, ct);
                    return document is null
                        ? Results.NotFound()
                        : Results.Ok(new DocumentStatus(
                            document.DocumentId, "received", document.ReceivedAt));
                })
            .WithName("DocumentStatus")
            .WithTags("Documents");

        app.MapPost(
                "/api/validate",
                (ValidatePropertyRequest request, PropertyValidator validator) =>
                    Results.Ok(validator.Validate(request)))
            .WithName("ValidateProperty")
            .WithTags("Validation");
    }
}
