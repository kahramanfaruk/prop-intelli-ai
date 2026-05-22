# Multi-stage build for the C# ingestion microservice.
# Build context is the `services/` directory (see docker-compose.yml).
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY IngestionApi/IngestionApi.csproj IngestionApi/
RUN dotnet restore IngestionApi/IngestionApi.csproj
COPY IngestionApi/ IngestionApi/
RUN dotnet publish IngestionApi/IngestionApi.csproj -c Release -o /app /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app .
EXPOSE 8080
ENV ASPNETCORE_URLS=http://+:8080
ENV ASPNETCORE_ENVIRONMENT=Production
ENTRYPOINT ["dotnet", "IngestionApi.dll"]
