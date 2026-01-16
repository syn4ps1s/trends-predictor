"""
Integration module - Connect trend research with demand prediction pipeline.

Workflow:
1. Get demand forecasts from DeepAR+
2. Research trends
3. Generate design concepts
4. Match designs with demand predictions
5. Recommend production plan
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TrendDemandIntegrator:
    """
    Integrate trend research with demand prediction pipeline.

    This module bridges:
    - TrendResearcher (trend discovery and design generation)
    - DeepARPipeline (demand forecasting)
    - MDPOptimizer (inventory optimization)

    Output: Actionable product recommendations with:
    - Design specifications (color, material, silhouette)
    - Forecasted demand (units)
    - Optimal production quantities
    - Pricing strategy
    - Timeline to market
    """

    def __init__(
        self,
        trend_researcher,
        deepar_pipeline,
        mdp_optimizer,
        store_aggregator,
    ):
        """
        Initialize integrator.

        Args:
            trend_researcher: TrendResearcher instance
            deepar_pipeline: Trained DeepAR+ pipeline
            mdp_optimizer: Trained MDP optimizer
            store_aggregator: Store aggregation module
        """
        self.trend_researcher = trend_researcher
        self.deepar = deepar_pipeline
        self.mdp = mdp_optimizer
        self.store_aggregator = store_aggregator

        logger.info("Initialized TrendDemandIntegrator")

    def create_collection_plan(
        self,
        season: str,
        X_forecast: np.ndarray,
        metadata: pd.DataFrame,
        num_designs: int = 20,
    ) -> Dict:
        """
        Create complete collection plan by integrating trends and demand.

        Args:
            season: Target season
            X_forecast: Feature sequences for demand prediction
            metadata: Metadata about data
            num_designs: Number of designs in collection

        Returns:
            Complete collection plan with designs, quantities, and timelines
        """
        logger.info(
            f"Creating {season.upper()} collection plan "
            f"({num_designs} designs)..."
        )

        # Step 1: Research trends for season
        logger.info("Step 1: Researching trends...")
        trend_insights = self.trend_researcher.research_trends(
            categories=["color", "material", "silhouette"],
            regions=["North America", "Europe", "Asia"],
            forecast_weeks=16,
        )

        # Step 2: Generate demand forecast
        logger.info("Step 2: Generating demand forecast...")
        demand_forecast = self._forecast_demand(X_forecast)

        # Step 3: Generate design concepts
        logger.info("Step 3: Generating design concepts...")
        concepts = self.trend_researcher.generate_design_concepts(
            trend_insights=trend_insights,
            num_concepts=num_designs,
            style=season,
        )

        # Step 4: Get design recommendations
        logger.info("Step 4: Matching designs with demand...")
        design_recs = self.trend_researcher.recommend_designs(
            design_concepts=concepts,
            demand_forecast=demand_forecast,
            num_recommendations=num_designs,
        )

        # Step 5: Optimize inventory with MDP
        logger.info("Step 5: Optimizing inventory...")
        inventory_plan = self._optimize_inventory_plan(design_recs, demand_forecast)

        # Step 6: Create production schedule
        logger.info("Step 6: Creating production schedule...")
        production_schedule = self._create_production_schedule(
            design_recs, inventory_plan, season
        )

        # Create final collection plan
        collection_plan = {
            "season": season,
            "creation_date": datetime.now().isoformat(),
            "num_designs": len(design_recs),
            "total_units_planned": int(
                sum(r.predicted_demand for r in design_recs)
            ),
            "trend_foundation": {
                "colors": trend_insights.get("color", []),
                "materials": trend_insights.get("material", []),
                "silhouettes": trend_insights.get("silhouette", []),
            },
            "design_recommendations": [
                self._serialize_design_rec(rec) for rec in design_recs
            ],
            "inventory_plan": inventory_plan,
            "production_schedule": production_schedule,
            "financial_summary": self._calculate_financial_summary(
                design_recs, inventory_plan
            ),
            "risk_assessment": self._assess_collection_risk(design_recs),
        }

        logger.info("Collection plan creation complete")

        return collection_plan

    def match_demand_to_trends(
        self,
        demand_forecast: pd.DataFrame,
        trend_insights: Dict,
    ) -> pd.DataFrame:
        """
        Match demand forecast with trend insights.

        Shows which predicted demand is driven by which trends.

        Args:
            demand_forecast: Demand predictions from DeepAR+
            trend_insights: Trend research results

        Returns:
            Demand forecast with trend drivers attached
        """
        logger.info("Matching demand to trends...")

        demand_copy = demand_forecast.copy()

        # Add trend driver columns
        demand_copy["trend_driver_color"] = None
        demand_copy["trend_driver_silhouette"] = None
        demand_copy["trend_driver_material"] = None
        demand_copy["confidence_score"] = 0.0

        # For each demand prediction, assign likely trends
        for idx, row in demand_copy.iterrows():
            color_trend = self._get_likely_trend(
                trend_insights.get("color", []), index=idx
            )
            silhouette_trend = self._get_likely_trend(
                trend_insights.get("silhouette", []), index=idx
            )
            material_trend = self._get_likely_trend(
                trend_insights.get("material", []), index=idx
            )

            demand_copy.loc[idx, "trend_driver_color"] = color_trend
            demand_copy.loc[idx, "trend_driver_silhouette"] = silhouette_trend
            demand_copy.loc[idx, "trend_driver_material"] = material_trend

            # Confidence based on trend consensus
            avg_confidence = np.mean([
                getattr(t, 'confidence_score', 0.5)
                for t in [color_trend, silhouette_trend, material_trend]
                if t
            ])
            demand_copy.loc[idx, "confidence_score"] = avg_confidence

        logger.info("Demand-trend matching complete")

        return demand_copy

    def recommend_promotional_strategy(
        self,
        design_recs: List,
        trend_insights: Dict,
    ) -> Dict:
        """
        Recommend promotional and marketing strategy.

        Args:
            design_recs: Design recommendations
            trend_insights: Trend insights

        Returns:
            Marketing and promotion strategy
        """
        logger.info("Creating promotional strategy...")

        strategy = {
            "campaign_themes": self._identify_campaign_themes(
                design_recs, trend_insights
            ),
            "influencer_partnerships": self._recommend_influencers(design_recs),
            "social_media_strategy": {
                "platforms": ["TikTok", "Instagram", "Pinterest"],
                "content_pillars": [
                    "Behind-the-scenes design process",
                    "Trend storytelling",
                    "Sustainability messaging",
                    "Styling inspiration",
                ],
                "posting_frequency": "4-5 times per week",
            },
            "email_marketing": {
                "segments": ["eco-conscious", "trendsetters", "classic_lovers"],
                "early_access": "48 hours before public launch",
                "personalization": "Based on past purchase patterns",
            },
            "pr_angle": self._create_pr_angle(design_recs, trend_insights),
            "launch_events": self._plan_launch_events(design_recs),
        }

        return strategy

    # ========== Private Methods ==========

    def _forecast_demand(self, X_forecast: np.ndarray) -> pd.DataFrame:
        """Generate demand forecast using DeepAR+."""
        predictions = self.deepar.predict(X_forecast, return_intervals=True)

        forecast_df = pd.DataFrame({
            "point_forecast": predictions["point_forecast"][:, 0],
            "lower_bound": predictions["lower_bound"][:, 0],
            "upper_bound": predictions["upper_bound"][:, 0],
            "forecast_date": [
                datetime.now() + timedelta(days=i) for i in range(len(predictions["point_forecast"]))
            ],
        })

        return forecast_df

    def _optimize_inventory_plan(
        self, design_recs: List, demand_forecast: pd.DataFrame
    ) -> Dict:
        """Optimize inventory using MDP."""
        plan = {}

        for rec in design_recs:
            # Get optimal order from MDP
            optimal_qty = self.mdp.get_optimal_order(
                current_inventory=0,
                predicted_demand_state="medium",
            )

            # Scale by predicted demand
            planned_qty = int(rec.predicted_demand * 0.8)  # 80% of forecast

            plan[rec.product_id] = {
                "design": rec.design_description[:50],
                "planned_production": planned_qty,
                "optimal_safety_stock": optimal_qty,
                "reorder_point": int(planned_qty * 0.3),
                "color": rec.primary_color,
                "material": rec.material,
            }

        return plan

    def _create_production_schedule(
        self,
        design_recs: List,
        inventory_plan: Dict,
        season: str,
    ) -> Dict:
        """Create production timeline."""
        # Calculate launch date (8 weeks before peak season)
        season_starts = {
            "spring": datetime(datetime.now().year, 3, 20),
            "summer": datetime(datetime.now().year, 6, 21),
            "fall": datetime(datetime.now().year, 9, 21),
            "winter": datetime(datetime.now().year, 12, 21),
        }

        target_launch = season_starts[season] - timedelta(weeks=8)

        # Group by material lead time
        schedule = {}

        for i, rec in enumerate(design_recs):
            lead_time = rec.lead_time_days
            production_start = target_launch - timedelta(days=lead_time)

            schedule[rec.product_id] = {
                "design_finalization": target_launch - timedelta(days=lead_time + 30),
                "material_procurement": production_start,
                "sample_production": production_start + timedelta(days=10),
                "full_production": production_start + timedelta(days=20),
                "quality_check": target_launch - timedelta(days=5),
                "warehouse_arrival": target_launch,
                "launch_date": target_launch,
            }

        return schedule

    def _calculate_financial_summary(
        self, design_recs: List, inventory_plan: Dict
    ) -> Dict:
        """Calculate financial projections."""
        total_revenue = sum(
            rec.predicted_demand * rec.target_price for rec in design_recs
        )

        total_cogs = sum(
            rec.predicted_demand * (rec.target_price * (1 - rec.estimated_margin))
            for rec in design_recs
        )

        gross_profit = total_revenue - total_cogs

        return {
            "total_revenue": float(round(total_revenue, 2)),
            "total_cogs": float(round(total_cogs, 2)),
            "gross_profit": float(round(gross_profit, 2)),
            "gross_margin": float(round(gross_profit / total_revenue, 3)),
            "units_planned": int(
                sum(inventory_plan[k]["planned_production"] for k in inventory_plan)
            ),
            "avg_selling_price": float(
                round(
                    np.mean([rec.target_price for rec in design_recs]), 2
                )
            ),
        }

    def _assess_collection_risk(self, design_recs: List) -> Dict:
        """Assess collection risk."""
        confidences = [rec.confidence for rec in design_recs]

        return {
            "confidence_range": (min(confidences), max(confidences)),
            "avg_confidence": float(np.mean(confidences)),
            "low_confidence_designs": sum(1 for c in confidences if c < 0.7),
            "high_confidence_designs": sum(1 for c in confidences if c > 0.85),
            "diversification_score": self._calculate_diversification(design_recs),
            "risk_assessment": "Low risk" if np.mean(confidences) > 0.75 else "Medium risk",
        }

    def _serialize_design_rec(self, rec) -> Dict:
        """Serialize design recommendation."""
        return {
            "product_id": rec.product_id,
            "category": rec.category,
            "design": rec.design_description,
            "primary_color": rec.primary_color,
            "secondary_colors": rec.secondary_colors,
            "material": rec.material,
            "silhouette": rec.silhouette,
            "predicted_demand": float(rec.predicted_demand),
            "confidence": float(rec.confidence),
            "target_price": float(rec.target_price),
            "estimated_margin": float(rec.estimated_margin),
            "lead_time_days": rec.lead_time_days,
        }

    def _get_likely_trend(self, trends: List, index: int) -> Optional[str]:
        """Get likely trend for index."""
        if not trends:
            return None

        trend_idx = index % len(trends)
        trend = trends[trend_idx]

        return getattr(trend, 'trend_name', None) if trend else None

    def _identify_campaign_themes(
        self, design_recs: List, trend_insights: Dict
    ) -> List[str]:
        """Identify campaign themes."""
        return [
            "Sustainable luxury",
            "Future of fashion",
            "Trend-forward classics",
            "Inclusive style",
        ]

    def _recommend_influencers(self, design_recs: List) -> Dict:
        """Recommend influencer partnerships."""
        return {
            "micro_influencers": 15,
            "macro_influencers": 3,
            "budget_allocation": "70% micro, 30% macro",
            "target_reach": "5M impressions minimum",
        }

    def _create_pr_angle(self, design_recs: List, trend_insights: Dict) -> str:
        """Create PR angle."""
        return (
            "AI-Powered Design Innovation: How Trend Intelligence and "
            "Demand Forecasting Shape the Future of Fashion"
        )

    def _plan_launch_events(self, design_recs: List) -> List[str]:
        """Plan launch events."""
        return [
            "Virtual launch event (livestream)",
            "In-store pop-up experiences (5 cities)",
            "Influencer collaboration events",
            "Media preview 2 weeks before launch",
        ]

    def _calculate_diversification(self, design_recs: List) -> float:
        """Calculate design diversification score."""
        colors = set(rec.primary_color for rec in design_recs)
        materials = set(rec.material for rec in design_recs)
        silhouettes = set(rec.silhouette for rec in design_recs)

        diversity = (
            len(colors) / 10 * 0.4
            + len(materials) / 5 * 0.3
            + len(silhouettes) / 5 * 0.3
        )

        return min(1.0, float(diversity))
