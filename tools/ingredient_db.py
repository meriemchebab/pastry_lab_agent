"""ingredient : hydration, fat, and solids."""
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class NutrientProfile:
    water: float
    fat: float
    solids: float

# Verified food-science composition standards
INGREDIENT_DATABASE: Dict[str, NutrientProfile] = {
    "butter_standard": NutrientProfile(water=0.16, fat=0.82, solids=0.02),
    "neutral_oil": NutrientProfile(water=0.00, fat=1.00, solids=0.00),
    "water": NutrientProfile(water=1.00, fat=0.00, solids=0.00),
    "whole_milk": NutrientProfile(water=0.88, fat=0.035, solids=0.085),
    "whole_egg": NutrientProfile(water=0.75, fat=0.10, solids=0.15),
    "granulated_sugar": NutrientProfile(water=0.00, fat=0.00, solids=1.00),
}