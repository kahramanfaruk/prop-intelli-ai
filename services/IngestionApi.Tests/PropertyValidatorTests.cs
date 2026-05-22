using IngestionApi.Models;
using IngestionApi.Services;
using Xunit;

namespace IngestionApi.Tests;

public class PropertyValidatorTests
{
    private readonly PropertyValidator _validator = new();

    [Fact]
    public void CompletePayloadIsValid()
    {
        var request = new ValidatePropertyRequest
        {
            PriceEur = 450_000m,
            LivingAreaSqm = 90,
            PostalCode = "90408",
            City = "Nürnberg",
            YearBuilt = 1998,
        };

        var result = _validator.Validate(request);

        Assert.True(result.IsValid);
        Assert.Empty(result.Findings);
    }

    [Fact]
    public void MissingMandatoryFieldsProduceErrors()
    {
        var result = _validator.Validate(new ValidatePropertyRequest { City = "Berlin" });

        Assert.False(result.IsValid);
        Assert.Contains(result.Findings, finding => finding.RuleId == "mandatory.price_eur");
        Assert.Contains(result.Findings, finding => finding.RuleId == "mandatory.living_area_sqm");
    }

    [Fact]
    public void NegativePriceIsRejected()
    {
        var request = new ValidatePropertyRequest
        {
            PriceEur = -5m,
            LivingAreaSqm = 90,
            PostalCode = "90408",
            City = "Nürnberg",
        };

        var result = _validator.Validate(request);

        Assert.False(result.IsValid);
        Assert.Contains(result.Findings, finding => finding.RuleId == "range.price_eur");
    }

    [Fact]
    public void InvalidPostalCodeWarnsButRemainsValid()
    {
        var request = new ValidatePropertyRequest
        {
            PriceEur = 450_000m,
            LivingAreaSqm = 90,
            PostalCode = "9040",
            City = "Nürnberg",
        };

        var result = _validator.Validate(request);

        Assert.True(result.IsValid); // a warning does not invalidate the payload
        Assert.Contains(result.Findings, finding => finding.RuleId == "format.postal_code");
    }
}
