"""Deterministic baking math: baker's percentages and pan geometry."""
import math
from langchain_core.tools import tool
from tools.ingredient_db import INGREDIENT_DATABASE

@tool
def scale_pan_geometry(source_shape: str, source_dim: float, target_shape: str, target_dim: float) -> str:
    """
    Calculates the scaling factor between two baking pans based on surface area.
    Use this when the user's pan size differs from the recipe.
    Args:
        source_shape: 'round' or 'square'
        source_dim: diameter or side length in inches
        target_shape: 'round' or 'square' 
        target_dim: diameter or side length in inches
    """
    area_src = math.pi * ((source_dim / 2) ** 2) if source_shape == "round" else source_dim ** 2
    area_tgt = math.pi * ((target_dim / 2) ** 2) if target_shape == "round" else target_dim ** 2
    scale_factor = round(area_tgt / area_src, 2)
    
    return f"Multiply all ingredients by {scale_factor} to properly scale the batter from a {source_dim}-inch {source_shape} pan to a {target_dim}-inch {target_shape} pan."

@tool
def calculate_fat_substitution(original_ingredient: str, original_grams: float, target_substitute: str) -> str:
    """
    Calculates exact substitute weight and fluid offset required to keep total fat and water content invariant.
    Ingredients must be standard keys like 'butter_standard', 'neutral_oil', 'whole_milk'////.
    """
    orig_key = original_ingredient.lower().replace(" ", "_")
    sub_key = target_substitute.lower().replace(" ", "_")

    if orig_key not in INGREDIENT_DATABASE or sub_key not in INGREDIENT_DATABASE:
        return f"Error: Ensure ingredients are exactly one of these keys: {list(INGREDIENT_DATABASE.keys())}."

    orig_prof = INGREDIENT_DATABASE[orig_key]
    sub_prof = INGREDIENT_DATABASE[sub_key]

    if sub_prof.fat <= 0:
        return f"Error: Target substitute '{target_substitute}' contains no fat."

    # Match total fat delivered by the original ingredient
    original_fat_mass = original_grams * orig_prof.fat
    substitute_grams = original_fat_mass / sub_prof.fat

    # Calculate moisture differential
    orig_water_mass = original_grams * orig_prof.water
    sub_water_mass = substitute_grams * sub_prof.water
    water_delta = orig_water_mass - sub_water_mass

    if water_delta > 0:
        return f"Use {substitute_grams:.1f}g of {target_substitute}. Add {water_delta:.1f}g of water or milk to replace missing moisture."
    elif water_delta < 0:
        return f"Use {substitute_grams:.1f}g of {target_substitute}. Withhold {abs(water_delta):.1f}g of liquid from the recipe to balance excess moisture."
    
    return f"Equal moisture profile. Use {substitute_grams:.1f}g of {target_substitute} directly."