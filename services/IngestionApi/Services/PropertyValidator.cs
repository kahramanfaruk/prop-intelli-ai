using System.Text.RegularExpressions;
using IngestionApi.Models;

namespace IngestionApi.Services;

/// <summary>
/// Plausibility validator that mirrors the Python validation rules, so the C#
/// boundary rejects the same implausible payloads the pipeline would flag.
/// </summary>
public sealed partial class PropertyValidator
{
    private const decimal PriceMax = 100_000_000m;
    private const double AreaMin = 5.0;
    private const double AreaMax = 1000.0;
    private const int YearMin = 1800;

    /// <summary>Validate a property payload and return findings plus an overall verdict.</summary>
    public ValidationResponse Validate(ValidatePropertyRequest request)
    {
        var findings = new List<ValidationFinding>();

        Mandatory(findings, "price_eur", request.PriceEur is not null);
        Mandatory(findings, "living_area_sqm", request.LivingAreaSqm is not null);
        Mandatory(findings, "postal_code", !string.IsNullOrWhiteSpace(request.PostalCode));
        Mandatory(findings, "city", !string.IsNullOrWhiteSpace(request.City));

        if (request.PriceEur is { } price && !(price > 0 && price < PriceMax))
        {
            findings.Add(new ValidationFinding(
                "range.price_eur", "price_eur", "error",
                $"Price {price} is outside the plausible range (0, {PriceMax})."));
        }

        if (request.LivingAreaSqm is { } area && !(area >= AreaMin && area <= AreaMax))
        {
            findings.Add(new ValidationFinding(
                "range.living_area_sqm", "living_area_sqm", "warning",
                $"Living area {area} m² is outside [{AreaMin}, {AreaMax}]."));
        }

        if (request.YearBuilt is { } year && !(year >= YearMin && year <= DateTime.UtcNow.Year))
        {
            findings.Add(new ValidationFinding(
                "range.year_built", "year_built", "warning",
                $"Construction year {year} is outside [{YearMin}, {DateTime.UtcNow.Year}]."));
        }

        if (!string.IsNullOrWhiteSpace(request.PostalCode) && !PostalCode().IsMatch(request.PostalCode))
        {
            findings.Add(new ValidationFinding(
                "format.postal_code", "postal_code", "warning",
                $"Postal code '{request.PostalCode}' is not a valid 5-digit German code."));
        }

        var isValid = findings.All(finding => finding.Severity != "error");
        return new ValidationResponse(isValid, findings);
    }

    private static void Mandatory(ICollection<ValidationFinding> findings, string field, bool present)
    {
        if (!present)
        {
            findings.Add(new ValidationFinding(
                $"mandatory.{field}", field, "error", $"Required field '{field}' is missing."));
        }
    }

    [GeneratedRegex("^[0-9]{5}$")]
    private static partial Regex PostalCode();
}
