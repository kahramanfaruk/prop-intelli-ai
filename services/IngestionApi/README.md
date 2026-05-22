# IngestionApi — C# / .NET 8 microservice

A lightweight ASP.NET Core minimal API that fronts the PropIntelli pipeline. It
accepts document uploads into the Bronze layer and exposes a validation endpoint
that mirrors the Python plausibility rules — so the same contract is enforced at
the service boundary and inside the pipeline.

## Why C# here

In a Microsoft-centric enterprise, the public ingestion/validation surface is a
natural fit for ASP.NET Core: first-class Azure integration (Container Apps, App
Service, Managed Identity), strong typing, and easy SDK access to Azure Blob /
Service Bus when this filesystem store is swapped for the cloud equivalents.

## Endpoints

| Method | Route                              | Purpose                                            |
| ------ | ---------------------------------- | -------------------------------------------------- |
| GET    | `/health`                          | Liveness probe.                                    |
| POST   | `/api/documents/upload`            | Multipart upload → Bronze store; returns a UUID.   |
| GET    | `/api/documents/{id}/status`       | Returns the stored document's manifest, or 404.    |
| POST   | `/api/validate`                    | Validates a property JSON payload (snake_case).    |

JSON is serialised as `snake_case` to match the Python schema contract. The
Bronze directory is resolved from `PROPINTELLI_DATA_DIR` (shared with the Python
services) or `Storage:DataDir` in `appsettings.json`.

## Run locally

```bash
dotnet run --project services/IngestionApi          # Swagger UI at /swagger
dotnet test services/IngestionApi.Tests             # unit + integration tests
```

## Example

```bash
curl -F "file=@sample_data/raw/expose_01_nuernberg_eigentumswohnung.pdf" \
     http://localhost:8080/api/documents/upload

curl -X POST http://localhost:8080/api/validate \
     -H 'Content-Type: application/json' \
     -d '{"price_eur":450000,"living_area_sqm":90,"postal_code":"90408","city":"Nürnberg"}'
```

## Production mapping

- Filesystem Bronze store → **Azure Blob Storage** (+ Event Grid to trigger the
  Python extraction worker).
- Service hosting → **Azure Container Apps** / App Service with Managed Identity.
- Validation contract → reused by Power Platform / Logic Apps front-ends.
