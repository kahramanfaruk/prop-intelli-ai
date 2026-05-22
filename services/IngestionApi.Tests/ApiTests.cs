using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace IngestionApi.Tests;

public class ApiTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ApiTests(WebApplicationFactory<Program> factory)
    {
        // Isolate the Bronze store under a temp directory for the test run.
        Environment.SetEnvironmentVariable(
            "PROPINTELLI_DATA_DIR",
            Path.Combine(Path.GetTempPath(), "propintelli-tests", Guid.NewGuid().ToString("N")));
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task HealthReturnsHealthy()
    {
        var response = await _client.GetAsync("/health");

        response.EnsureSuccessStatusCode();
        using var document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("healthy", document.RootElement.GetProperty("status").GetString());
    }

    [Fact]
    public async Task ValidateAcceptsSnakeCasePayload()
    {
        const string payload =
            """{"price_eur":450000,"living_area_sqm":90,"postal_code":"90408","city":"Nürnberg"}""";
        var response = await _client.PostAsync(
            "/api/validate", new StringContent(payload, Encoding.UTF8, "application/json"));

        response.EnsureSuccessStatusCode();
        using var document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.True(document.RootElement.GetProperty("is_valid").GetBoolean());
    }

    [Fact]
    public async Task ValidateFlagsMissingMandatoryFields()
    {
        var response = await _client.PostAsync(
            "/api/validate", new StringContent("""{"city":"Berlin"}""", Encoding.UTF8, "application/json"));

        response.EnsureSuccessStatusCode();
        using var document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.False(document.RootElement.GetProperty("is_valid").GetBoolean());
    }

    [Fact]
    public async Task UploadStoresDocumentAndExposesStatus()
    {
        using var content = new MultipartFormDataContent();
        var fileBytes = Encoding.UTF8.GetBytes("%PDF-1.4 minimal test payload");
        var fileContent = new ByteArrayContent(fileBytes);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/pdf");
        content.Add(fileContent, "file", "test.pdf");

        var upload = await _client.PostAsync("/api/documents/upload", content);
        upload.EnsureSuccessStatusCode();
        using var uploaded = JsonDocument.Parse(await upload.Content.ReadAsStringAsync());
        var documentId = uploaded.RootElement.GetProperty("document_id").GetString();
        Assert.False(string.IsNullOrEmpty(documentId));

        var status = await _client.GetAsync($"/api/documents/{documentId}/status");
        status.EnsureSuccessStatusCode();

        var missing = await _client.GetAsync("/api/documents/does-not-exist/status");
        Assert.Equal(HttpStatusCode.NotFound, missing.StatusCode);
    }
}
