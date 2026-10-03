"""Unit tests for pan conversions and fat balance."""

import pytest

from tools.baking_math import calculate_fat_substitution, scale_pan_geometry


def test_same_round_pan_has_no_scaling():
    result = scale_pan_geometry.invoke(
        {"source_shape": "round", "source_dim": 8, "target_shape": "round", "target_dim": 8}
    )

    assert "Multiply all ingredients by 1.0" in result
    assert "8-centimeters round pan" in result


def test_round_pan_scaling_uses_area_ratio():
    result = scale_pan_geometry.invoke(
        {"source_shape": "round", "source_dim": 8, "target_shape": "round", "target_dim": 10}
    )

    assert "Multiply all ingredients by 1.56" in result


def test_square_to_round_pan_scaling():
    result = scale_pan_geometry.invoke(
        {"source_shape": "square", "source_dim": 8, "target_shape": "round", "target_dim": 8}
    )

    assert "Multiply all ingredients by 0.79" in result


def test_centimeter_pan_scaling():
    result = scale_pan_geometry.invoke(
        {"source_shape": "round", "source_dim": 20, "target_shape": "round", "target_dim": 25, "unit": "cm"}
    )

    assert "Multiply all ingredients by 1.56" in result
    assert "20-centimeters round pan" in result


def test_dimensions_scale_consistently_in_inches():
    source_inches = 8
    target_inches = 10
    result = scale_pan_geometry.invoke(
        {"source_shape": "round", "source_dim": source_inches, "target_shape": "round", "target_dim": target_inches, "unit": "inches"}
    )

    assert "Multiply all ingredients by 1.56" in result
    assert "8-inches round pan" in result


def test_invalid_pan_unit_returns_error():
    result = scale_pan_geometry.invoke(
        {"source_shape": "round", "source_dim": 8, "target_shape": "round", "target_dim": 10, "unit": "millimeters"}
    )

    assert "unit must be 'inches' or 'centimeters'" in result


def test_zero_source_dimension_raises_division_error():
    with pytest.raises(ZeroDivisionError):
        scale_pan_geometry.invoke(
            {"source_shape": "round", "source_dim": 0, "target_shape": "square", "target_dim": 8}
        )


def test_unrecognized_pan_shape_currently_uses_square_area():
    """Document current behavior for pan shapes the tool does not validate."""
    result = scale_pan_geometry.invoke(
        {"source_shape": "triangle", "source_dim": 8, "target_shape": "square", "target_dim": 8}
    )

    assert "Multiply all ingredients by 1.0" in result


def test_butter_to_oil_matches_fat_and_reports_missing_moisture():
    result = calculate_fat_substitution.invoke(
        {
            "original_ingredient": "butter_standard",
            "original_grams": 100,
            "target_substitute": "neutral_oil",
        }
    )

    assert "Use 82.0g of neutral_oil" in result
    assert "Add 16.0g of water or milk" in result


def test_oil_to_butter_matches_fat_and_reports_excess_moisture():
    result = calculate_fat_substitution.invoke(
        {
            "original_ingredient": "neutral_oil",
            "original_grams": 100,
            "target_substitute": "butter_standard",
        }
    )

    assert "Use 122.0g of butter_standard" in result
    assert "Withhold 19.5g of liquid" in result


def test_equal_water_profiles_do_not_need_liquid_adjustment():
    result = calculate_fat_substitution.invoke(
        {
            "original_ingredient": "neutral_oil",
            "original_grams": 50,
            "target_substitute": "neutral_oil",
        }
    )

    assert "Equal moisture profile" in result
    assert "Use 50.0g of neutral_oil directly" in result


def test_unknown_ingredient_returns_available_keys():
    result = calculate_fat_substitution.invoke(
        {
            "original_ingredient": "margarine",
            "original_grams": 100,
            "target_substitute": "neutral_oil",
        }
    )

    assert result.startswith("Error: Ensure ingredients are exactly one of these keys:")
    assert "butter_standard" in result


def test_zero_fat_substitute_returns_error():
    result = calculate_fat_substitution.invoke(
        {
            "original_ingredient": "butter_standard",
            "original_grams": 100,
            "target_substitute": "granulated_sugar",
        }
    )

    assert "contains no fat" in result


def test_negative_original_weight_currently_produces_negative_substitute():
    """Document current behavior for weights the tool does not validate."""
    result = calculate_fat_substitution.invoke(
        {
            "original_ingredient": "butter_standard",
            "original_grams": -100,
            "target_substitute": "neutral_oil",
        }
    )

    assert "Use -82.0g of neutral_oil" in result
