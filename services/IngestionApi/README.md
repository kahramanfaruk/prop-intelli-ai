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

### Upload validation & limits

`/api/documents/upload` is hardened against malformed and abusive input:

- **PDF only** — the payload must begin with the `%PDF-` signature (magic bytes),
  checked on the buffered content; a non-PDF is rejected with `400` regardless of
  the declared content type.
- **Size cap** — bounded by `Upload:MaxBytes` (default 25 MB), enforced both at
  the framework's multipart limit and with an explicit `413` from the endpoint.

Authentication is intentionally omitted for this take-home; in production the
surface sits behind **Azure API Management + Microsoft Entra ID** (see the mapping
below).

### How an upload reaches the pipeline

The API only ingests into Bronze; extraction is decoupled. The Python worker
(`propintelli watch`, run by the `worker` service in `docker compose`) polls the
shared Bronze store and processes any newly-ingested document into the Silver
store. So `POST /api/documents/upload` → Bronze → worker → Silver, end to end,
without the API and the extractor being directly coupled. (In production the poll
becomes a Blob → Event Grid → Service Bus trigger.)

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
