using System.Text.Json;
using IngestionApi.Endpoints;
using IngestionApi.Services;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);

// snake_case JSON for parity with the Python schema contract.
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
});

// Cap multipart uploads at the framework level so an oversized body is rejected
// before it is buffered; the endpoint additionally returns a clear 413.
var maxUploadBytes = builder.Configuration.GetValue<long?>("Upload:MaxBytes") ?? 26_214_400L;
builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = maxUploadBytes;
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
