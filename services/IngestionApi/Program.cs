using System.Text.Json;
using IngestionApi.Endpoints;
using IngestionApi.Services;

var builder = WebApplication.CreateBuilder(args);

// snake_case JSON for parity with the Python schema contract.
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
});

// Cap the request body at the server level: Kestrel rejects an oversized upload
// with 413 before it is buffered. The endpoint repeats the size check (on the
// file part) for hosts that bypass this limit, e.g. the in-memory TestServer.
var maxUploadBytes = builder.Configuration.GetValue<long?>("Upload:MaxBytes") ?? 26_214_400L;
builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = maxUploadBytes;
});

builder.Services.AddSingleton<FileDocumentStore>();
builder.Services.AddSingleton<PropertyValidator>();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapApiEndpoints();
app.Run();

/// <summary>Entry-point class exposed for the integration test host.</summary>
public partial class Program { }
