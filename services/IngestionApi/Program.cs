using System.Text.Json;
using IngestionApi.Endpoints;
using IngestionApi.Services;

var builder = WebApplication.CreateBuilder(args);

// snake_case JSON for parity with the Python schema contract.
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
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
