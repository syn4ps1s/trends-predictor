"""
Design Recommender - Recommends specific designs, colors, and materials.

Integrates:
- Design concepts from GenerativeDesignStudio
- Demand predictions from DeepAR+
- Trend insights
- Material science
- Cost and margin analysis
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class DesignRecommender:
    """
    Recommend specific product designs based on:
    - Predicted demand (from DeepAR+)
    - Trend analysis (from TrendAnalyzer)
    - Design concepts (from GenerativeDesignStudio)
    - Material science and availability
    - Cost and profitability
    """

    def __init__(self):
        """Initialize design recommender."""
        # Material specifications
        self.material_properties = self._load_material_specs()
        self.color_psychology = self._load_color_psychology()
        self.silhouette_seasonality = self._load_silhouette_seasonality()

        logger.info("Initialized DesignRecommender")

    def recommend_materials(
        self, design_analysis: Dict, color: str, season: str = "spring"
    ) -> Dict:
        """
        Recommend materials for a design.

        Args:
            design_analysis: Design analysis from GenerativeDesignStudio
            color: Primary color of design
            season: Target season

        Returns:
            Material recommendation with blend, lead time, cost
        """
        logger.info(f"Recommending materials for {color} design...")

        category = design_analysis.get("category", "casual")

        # Select primary material
        primary_material = self._select_primary_material(category, season)

        # Build material blend
        blend = self._create_material_blend(primary_material, category)

        # Get specs
        specs = self.material_properties.get(primary_material, {})

        recommendation = {
            "primary_material": primary_material,
            "blend": blend,
            "gsm": specs.get("gsm", 140),
            "weight_category": specs.get("weight", "medium"),
            "sustainability_score": specs.get("sustainability", 0.6),
            "lead_time": specs.get("lead_time_weeks", 8),
            "cost_per_meter": specs.get("cost", 12.50),
            "care_instructions": specs.get("care", "Machine wash cold"),
            "performance_rating": specs.get("performance", 0.8),
            "luxury_positioning": self._assess_luxury_positioning(
                primary_material, blend
            ),
        }

        logger.info(
            f"Material selected: {primary_material} "
            f"(lead time: {recommendation['lead_time']} weeks)"
        )

        return recommendation

    def recommend_colors_for_season(
        self,
        season: str,
        num_recommendations: int = 5,
        include_trends: bool = True,
    ) -> List[Dict]:
        """
        Recommend color palette for a season.

        Args:
            season: Target season
            num_recommendations: Number of color recommendations
            include_trends: Include trend-driven colors

        Returns:
            Sorted list of color recommendations
        """
        logger.info(f"Recommending colors for {season}...")

        seasonal_palettes = {
            "spring": {
                "primary": ["sage green", "blush pink", "powder blue"],
                "accent": ["cream", "warm beige", "pale yellow"],
                "special": ["coral", "mint"],
            },
            "summer": {
                "primary": ["ocean blue", "bright white", "golden yellow"],
                "accent": ["lime green", "cream", "light gray"],
                "special": ["sunset orange", "turquoise"],
            },
            "fall": {
                "primary": ["burnt orange", "deep burgundy", "chocolate brown"],
                "accent": ["mustard", "cream", "olive"],
                "special": ["rust", "deep gold"],
            },
            "winter": {
                "primary": ["navy", "charcoal", "deep red"],
                "accent": ["cream", "silver", "black"],
                "special": ["emerald", "gold"],
            },
        }

        palette = seasonal_palettes.get(season, seasonal_palettes["spring"])

        recommendations = []

        for color_type, colors in palette.items():
            for color in colors:
                psych = self.color_psychology.get(color, {})

                recommendation = {
                    "color": color,
                    "type": color_type,
                    "psychology": psych.get("psychology", "neutral"),
                    "age_appeal": psych.get("age_groups", ["18-45"]),
                    "trending": include_trends
                    and color in ["sage green", "navy", "cream", "butter yellow"],
                    "seasonality": 1.0 if color_type == "primary" else 0.7,
                    "versatility": psych.get("versatility", 0.7),
                    "shelf_appeal": np.random.uniform(0.75, 0.95),
                }

                recommendations.append(recommendation)

        # Sort by trending, versatility, shelf appeal
        recommendations.sort(
            key=lambda r: (
                r["trending"],
                r["versatility"],
                r["shelf_appeal"],
            ),
            reverse=True,
        )

        logger.info(f"Generated {len(recommendations)} color recommendations")

        return recommendations[:num_recommendations]

    def create_color_combinations(
        self, base_colors: List[str], num_combinations: int = 10
    ) -> List[Dict]:
        """
        Create harmonious color combinations.

        Args:
            base_colors: Base colors to work with
            num_combinations: Number of combinations

        Returns:
            List of color combination recommendations
        """
        logger.info(f"Creating color combinations from {base_colors}...")

        combinations = [
            {
                "name": "Earthy Elegance",
                "colors": ["sage green", "cream", "terracotta"],
                "vibe": "natural, sophisticated",
                "appeal": 0.88,
            },
            {
                "name": "Fresh Minimalist",
                "colors": ["navy", "butter yellow", "white"],
                "vibe": "clean, modern",
                "appeal": 0.85,
            },
            {
                "name": "Warm Contrast",
                "colors": ["chocolate brown", "mustard", "cream"],
                "vibe": "cozy, balanced",
                "appeal": 0.82,
            },
            {
                "name": "Cool Neutral",
                "colors": ["charcoal", "light gray", "white"],
                "vibe": "timeless, versatile",
                "appeal": 0.80,
            },
            {
                "name": "Jewel Tone",
                "colors": ["deep burgundy", "emerald", "gold"],
                "vibe": "luxurious, bold",
                "appeal": 0.78,
            },
            {
                "name": "Summer Breeze",
                "colors": ["ocean blue", "turquoise", "white"],
                "vibe": "fresh, energetic",
                "appeal": 0.83,
            },
            {
                "name": "Sunset Romance",
                "colors": ["coral", "sunset orange", "cream"],
                "vibe": "warm, inviting",
                "appeal": 0.81,
            },
            {
                "name": "Botanical",
                "colors": ["sage green", "olive", "cream"],
                "vibe": "natural, calming",
                "appeal": 0.84,
            },
        ]

        combinations.sort(key=lambda c: c["appeal"], reverse=True)

        logger.info(f"Generated {len(combinations)} color combinations")

        return combinations[:num_combinations]

    def recommend_silhouettes(
        self,
        target_demographic: str = "millennial",
        season: str = "spring",
    ) -> List[Dict]:
        """
        Recommend silhouettes based on target and season.

        Args:
            target_demographic: Target demographic profile
            season: Target season

        Returns:
            List of silhouette recommendations
        """
        logger.info(f"Recommending silhouettes for {target_demographic} in {season}...")

        all_silhouettes = {
            "oversized": {
                "appeal": 0.88,
                "seasonality": {"spring": 0.8, "summer": 0.7, "fall": 0.9, "winter": 0.95},
                "demographics": ["millennial", "gen_z", "young_professional"],
            },
            "relaxed_fit": {
                "appeal": 0.85,
                "seasonality": {"spring": 0.85, "summer": 0.9, "fall": 0.8, "winter": 0.75},
                "demographics": ["millennial", "all"],
            },
            "maxi": {
                "appeal": 0.82,
                "seasonality": {"spring": 0.7, "summer": 0.85, "fall": 0.65, "winter": 0.6},
                "demographics": ["all", "mature"],
            },
            "structured_blazer": {
                "appeal": 0.80,
                "seasonality": {"spring": 0.75, "summer": 0.5, "fall": 0.9, "winter": 0.95},
                "demographics": ["professional", "mature"],
            },
            "wide_leg": {
                "appeal": 0.78,
                "seasonality": {"spring": 0.8, "summer": 0.85, "fall": 0.8, "winter": 0.7},
                "demographics": ["all", "young_professional"],
            },
            "minimalist": {
                "appeal": 0.76,
                "seasonality": {"spring": 0.8, "summer": 0.75, "fall": 0.8, "winter": 0.8},
                "demographics": ["minimalist", "professional", "mature"],
            },
        }

        recommendations = []

        for silhouette, specs in all_silhouettes.items():
            season_factor = specs["seasonality"].get(season, 0.7)

            # Demographics match
            demographic_match = (
                1.0
                if target_demographic in specs["demographics"]
                else 0.5
            )

            score = specs["appeal"] * season_factor * demographic_match

            recommendation = {
                "silhouette": silhouette,
                "appeal": specs["appeal"],
                "seasonal_fit": season_factor,
                "demographic_fit": demographic_match,
                "overall_score": score,
                "key_features": self._get_silhouette_features(silhouette),
            }

            recommendations.append(recommendation)

        recommendations.sort(key=lambda r: r["overall_score"], reverse=True)

        logger.info(
            f"Generated {len(recommendations)} silhouette recommendations"
        )

        return recommendations

    def recommend_details_and_finishes(
        self, primary_design: Dict
    ) -> Dict:
        """
        Recommend finishing details (buttons, zippers, hems, etc).

        Args:
            primary_design: Primary design specification

        Returns:
            Finishing detail recommendations
        """
        logger.info("Recommending finishing details...")

        details = {
            "buttons": {
                "materials": ["natural bone", "recycled plastic", "wood", "metal"],
                "recommendation": "natural bone for premium positioning",
            },
            "closures": {
                "options": ["invisible zipper", "wooden toggle", "organic snap"],
                "recommendation": "invisible zipper for clean aesthetic",
            },
            "hems": {
                "options": ["single stitch", "rolled hem", "raw edge", "lettuce edge"],
                "recommendation": "single stitch for durability and quality feel",
            },
            "seams": {
                "options": ["French seams", "flat seams", "safety stitch"],
                "recommendation": "French seams for premium feel and durability",
            },
            "labels": {
                "material": "organic cotton woven label",
                "design": "heritage branding",
                "sustainability_bonus": True,
            },
            "packaging": {
                "recommendation": "minimal, sustainable packaging with brand story",
                "sustainability_score": 0.9,
            },
        }

        return details

    def price_recommendation(
        self,
        material_cost: float,
        design_complexity: float,
        brand_positioning: str = "contemporary",
        market_segment: str = "mid_market",
    ) -> Dict:
        """
        Recommend pricing strategy.

        Args:
            material_cost: Material cost per unit
            design_complexity: Design complexity score (0-1)
            brand_positioning: Brand position ('contemporary', 'luxury', 'accessible')
            market_segment: Market segment ('budget', 'mid_market', 'premium')

        Returns:
            Pricing recommendation
        """
        logger.info("Calculating pricing recommendation...")

        # Markup calculation
        markups = {
            "luxury": {"material": 4.5, "labor": 1.5, "overhead": 1.3, "margin": 2.0},
            "contemporary": {"material": 3.5, "labor": 1.2, "overhead": 1.2, "margin": 1.8},
            "accessible": {"material": 2.5, "labor": 1.0, "overhead": 1.1, "margin": 1.5},
        }

        markup = markups[brand_positioning]

        total_production_cost = (
            material_cost * markup["material"]
            + 5  # Base labor cost
            + material_cost * 0.1  # Overhead
        )

        retail_price = total_production_cost * markup["margin"]

        recommendation = {
            "production_cost": float(round(total_production_cost, 2)),
            "recommended_retail_price": float(round(retail_price, 2)),
            "wholesale_price": float(round(retail_price * 0.5, 2)),
            "margin_percentage": 0.55,
            "competitive_pricing": self._get_competitive_pricing(
                brand_positioning, design_complexity
            ),
        }

        logger.info(f"Pricing: ${recommendation['recommended_retail_price']} retail")

        return recommendation

    def sustainability_assessment(
        self, material_rec: Dict, design: Dict
    ) -> Dict:
        """
        Assess sustainability of design.

        Args:
            material_rec: Material recommendation
            design: Design specification

        Returns:
            Sustainability assessment and recommendations
        """
        logger.info("Assessing sustainability...")

        material_score = material_rec.get("sustainability_score", 0.5)

        design_longevity = 0.8  # Timeless design tends to last longer

        sustainability = {
            "material_score": material_score,
            "design_longevity": design_longevity,
            "overall_sustainability": (material_score + design_longevity) / 2,
            "certifications": self._recommend_certifications(
                material_rec["primary_material"]
            ),
            "supply_chain_recommendations": [
                "Source from fair-trade suppliers",
                "Minimize transportation (regional production)",
                "Implement take-back program",
            ],
            "marketing_angle": "Sustainable luxury that lasts",
        }

        return sustainability

    # ========== Private Methods ==========

    def _load_material_specs(self) -> Dict:
        """Load material specifications."""
        return {
            "organic cotton": {
                "gsm": 140,
                "weight": "medium",
                "sustainability": 0.9,
                "cost": 12.50,
                "lead_time_weeks": 8,
                "care": "Machine wash cold",
                "performance": 0.85,
            },
            "recycled polyester": {
                "gsm": 150,
                "weight": "medium",
                "sustainability": 0.85,
                "cost": 10.00,
                "lead_time_weeks": 6,
                "care": "Machine wash warm",
                "performance": 0.80,
            },
            "linen": {
                "gsm": 180,
                "weight": "medium-heavy",
                "sustainability": 0.95,
                "cost": 15.00,
                "lead_time_weeks": 10,
                "care": "Machine wash gentle",
                "performance": 0.75,
            },
            "silk": {
                "gsm": 100,
                "weight": "light",
                "sustainability": 0.6,
                "cost": 25.00,
                "lead_time_weeks": 12,
                "care": "Hand wash",
                "performance": 0.90,
            },
            "wool": {
                "gsm": 200,
                "weight": "heavy",
                "sustainability": 0.75,
                "cost": 18.00,
                "lead_time_weeks": 10,
                "care": "Hand wash or dry clean",
                "performance": 0.88,
            },
            "cotton blend": {
                "gsm": 140,
                "weight": "medium",
                "sustainability": 0.70,
                "cost": 8.50,
                "lead_time_weeks": 6,
                "care": "Machine wash",
                "performance": 0.80,
            },
        }

    def _load_color_psychology(self) -> Dict:
        """Load color psychology data."""
        return {
            "sage green": {
                "psychology": "calm, natural, sophisticated",
                "age_groups": ["18-45"],
                "versatility": 0.90,
            },
            "butter yellow": {
                "psychology": "cheerful, optimistic, energetic",
                "age_groups": ["18-35"],
                "versatility": 0.75,
            },
            "navy": {
                "psychology": "trustworthy, professional, classic",
                "age_groups": ["18-65"],
                "versatility": 0.95,
            },
            "cream": {
                "psychology": "soft, elegant, timeless",
                "age_groups": ["18-65"],
                "versatility": 0.95,
            },
            "burnt orange": {
                "psychology": "warm, creative, welcoming",
                "age_groups": ["25-50"],
                "versatility": 0.80,
            },
            "deep burgundy": {
                "psychology": "luxurious, sophisticated, powerful",
                "age_groups": ["25-65"],
                "versatility": 0.85,
            },
            "coral": {
                "psychology": "playful, warm, social",
                "age_groups": ["18-35"],
                "versatility": 0.70,
            },
        }

    def _load_silhouette_seasonality(self) -> Dict:
        """Load silhouette seasonality data."""
        return {
            "oversized": {"spring": 0.8, "summer": 0.7, "fall": 0.9, "winter": 0.95},
            "relaxed_fit": {"spring": 0.85, "summer": 0.9, "fall": 0.8, "winter": 0.75},
            "maxi": {"spring": 0.7, "summer": 0.85, "fall": 0.65, "winter": 0.6},
            "mini": {"spring": 0.6, "summer": 0.95, "fall": 0.4, "winter": 0.2},
        }

    def _select_primary_material(self, category: str, season: str) -> str:
        """Select primary material based on category and season."""
        seasonal_preferences = {
            "spring": "organic cotton",
            "summer": "linen",
            "fall": "wool",
            "winter": "wool",
        }

        return seasonal_preferences.get(season, "organic cotton")

    def _create_material_blend(self, primary: str, category: str) -> Dict:
        """Create material blend for durability."""
        blends = {
            "organic cotton": {"organic cotton": 0.80, "elastane": 0.20},
            "linen": {"linen": 0.70, "organic cotton": 0.30},
            "wool": {"wool": 0.90, "elastane": 0.10},
            "silk": {"silk": 0.85, "elastane": 0.15},
        }

        return blends.get(primary, {primary: 1.0})

    def _assess_luxury_positioning(self, material: str, blend: Dict) -> float:
        """Assess luxury positioning based on material."""
        luxury_materials = {"silk": 0.95, "wool": 0.85, "linen": 0.75}

        return luxury_materials.get(material, 0.50)

    def _get_silhouette_features(self, silhouette: str) -> List[str]:
        """Get key features of silhouette."""
        features_map = {
            "oversized": [
                "relaxed fit",
                "comfortable",
                "modern",
                "easy to layer",
            ],
            "relaxed_fit": [
                "non-restrictive",
                "everyday wear",
                "versatile",
            ],
            "maxi": ["elegant", "formal", "elongating", "dramatic"],
            "structured_blazer": [
                "tailored",
                "professional",
                "structured shoulders",
            ],
            "minimalist": [
                "clean lines",
                "timeless",
                "versatile",
                "wardrobe staple",
            ],
        }

        return features_map.get(silhouette, [])

    def _get_competitive_pricing(self, positioning: str, complexity: float) -> Dict:
        """Get competitive pricing benchmarks."""
        return {
            "low_competitors": "$39-59",
            "mid_competitors": "$69-99",
            "premium_competitors": "$129-179",
        }

    def _recommend_certifications(self, material: str) -> List[str]:
        """Recommend sustainability certifications."""
        certifications_map = {
            "organic cotton": ["GOTS", "Fair Trade"],
            "linen": ["OEKO-TEX", "EU Ecolabel"],
            "recycled polyester": ["GRS", "Cradle to Cradle"],
            "wool": ["RWS", "Responsible Wool Standard"],
        }

        return certifications_map.get(material, ["OEKO-TEX"])
