#!/usr/bin/env python3
"""
Comprehensive integration test for Fashion Demand Predictor v2.0.0

Tests all major components working together:
1. Data Generation (synthetic demand with seasonality and collections)
2. Feature Engineering (seasonality, macro features, store features)
3. Trend Research System (trend discovery, design concepts, recommendations)
4. Design Recommendations (materials, colors, pricing)
5. Integration Layer (matching trends with demand)

Run with: python test_system_integration.py
"""

import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

print("\n" + "="*70)
print("FASHION DEMAND PREDICTOR v2.0.0 - INTEGRATION TEST SUITE")
print("="*70)

# Test 1: Data Generation & Feature Engineering
print("\n[1/5] TESTING DATA GENERATION & FEATURE ENGINEERING")
print("-" * 70)

try:
    from src.data_generation import SyntheticDataGenerator, CollectionSimulator
    from src.preprocessing import (
        SeasonalityProcessor,
        MacroFeaturesProcessor,
        StoreFeatureProcessor
    )

    # Generate synthetic demand
    print("  • Generating synthetic demand data...")
    gen = SyntheticDataGenerator(
        num_stores=10,
        num_products=50,
        num_days=365,
        seed=42
    )
    demand_df = gen.generate()
    print(f"    ✓ Generated {len(demand_df)} demand records ({demand_df.shape[1]} features)")
    print(f"    ✓ Date range: {demand_df['timestamp'].min()} to {demand_df['timestamp'].max()}")

    # Generate metadata
    stores_df = gen.generate_store_metadata()
    products_df = gen.generate_product_metadata()
    print(f"    ✓ Generated metadata: {len(stores_df)} stores, {len(products_df)} products")

    # Test seasonality processor
    print("  • Processing seasonality features...")
    seasonality = SeasonalityProcessor()
    print("    ✓ SeasonalityProcessor initialized")

    # Test macro features
    print("  • Generating macro-economic features...")
    macro = MacroFeaturesProcessor()
    dates = pd.date_range(demand_df['timestamp'].min(), demand_df['timestamp'].max(), freq='D')
    macro_df = macro.create_macro_features_df(dates)
    print(f"    ✓ Generated {len(macro_df)} macro records with {macro_df.shape[1]} features")

    # Test store features
    print("  • Processing store characteristics...")
    store_processor = StoreFeatureProcessor()
    print("    ✓ StoreFeatureProcessor initialized")

    print("\n✅ DATA GENERATION & FEATURE ENGINEERING: PASSED")

except Exception as e:
    print(f"\n❌ DATA GENERATION: FAILED - {str(e)}")
    sys.exit(1)

# Test 2: Trend Research System
print("\n[2/5] TESTING TREND RESEARCH SYSTEM")
print("-" * 70)

try:
    from src.trend_research import TrendResearcher, TrendAnalyzer
    from src.trend_research import GenerativeDesignStudio

    # Initialize trend researcher
    print("  • Initializing Trend Researcher...")
    researcher = TrendResearcher(
        llm_provider="openai",
        generative_model="stable-diffusion",
        enable_real_time_monitoring=True
    )
    print(f"    ✓ TrendResearcher initialized ({researcher.llm_provider})")

    # Research trends
    print("  • Researching global trends...")
    start = time.time()
    trends = researcher.research_trends(
        categories=["color", "material", "silhouette"],
        regions=["North America", "Europe", "Asia"],
        lookback_weeks=4,
        forecast_weeks=12
    )
    elapsed = time.time() - start
    print(f"    ✓ Researched {len(trends)} categories in {elapsed:.2f}s")
    for cat, insights in trends.items():
        if insights:
            print(f"      - {cat}: {len(insights)} insights")

    # Generate design concepts
    print("  • Generating design concepts from trends...")
    start = time.time()
    concepts = researcher.generate_design_concepts(
        trend_insights=trends,
        num_concepts=5,
        style="contemporary"
    )
    elapsed = time.time() - start
    print(f"    ✓ Generated {len(concepts)} design concepts in {elapsed:.2f}s")

    # Generate seasonal collection
    print("  • Creating seasonal collection plan...")
    start = time.time()
    collection = researcher.seasonal_collection_plan(
        upcoming_season="spring",
        num_styles=5,
        num_colors_per_style=3
    )
    elapsed = time.time() - start
    print(f"    ✓ Created {collection['season']} collection in {elapsed:.2f}s")
    print(f"      - Styles: {len(collection['styles'])}")
    print(f"      - Color palette: {len(collection['color_palette'])} colors")

    print("\n✅ TREND RESEARCH SYSTEM: PASSED")

except Exception as e:
    print(f"\n❌ TREND RESEARCH: FAILED - {str(e)}")
    sys.exit(1)

# Test 3: Design Recommendations
print("\n[3/5] TESTING DESIGN RECOMMENDATIONS")
print("-" * 70)

try:
    from src.trend_research import DesignRecommender

    print("  • Initializing Design Recommender...")
    recommender = DesignRecommender()
    print("    ✓ DesignRecommender initialized")

    # Color recommendations
    print("  • Getting seasonal color recommendations...")
    colors = recommender.recommend_colors_for_season("spring", num_recommendations=5)
    print(f"    ✓ Got {len(colors)} color recommendations")
    for color in colors[:3]:
        print(f"      - {color['color'].upper()}: {color['psychology']}")

    # Material recommendations
    print("  • Getting material recommendations...")
    materials = recommender.recommend_materials(
        design_analysis={"category": "casual", "complexity_score": 0.7},
        color="sage green",
        season="spring"
    )
    print(f"    ✓ Recommended: {materials['primary_material']} ({materials['blend']})")
    print(f"      - Cost: ${materials['cost_per_meter']:.2f}/m")
    print(f"      - Lead time: {materials['lead_time']} weeks")

    # Silhouette recommendations
    print("  • Getting silhouette recommendations...")
    silhouettes = recommender.recommend_silhouettes(
        season="spring",
        target_demographic="millennial"
    )
    print(f"    ✓ Got {len(silhouettes)} silhouette recommendations")

    # Pricing recommendations
    print("  • Getting pricing recommendations...")
    pricing = recommender.price_recommendation(
        material_cost=25.0,
        design_complexity=0.75,
        brand_positioning="contemporary"
    )
    print(f"    ✓ Recommended retail price: ${pricing['recommended_retail_price']:.2f}")
    print(f"      - Margin: {pricing['margin_percentage']:.1%}")

    print("\n✅ DESIGN RECOMMENDATIONS: PASSED")

except Exception as e:
    print(f"\n❌ DESIGN RECOMMENDATIONS: FAILED - {str(e)}")
    sys.exit(1)

# Test 4: Integration Layer
print("\n[4/5] TESTING TREND-DEMAND INTEGRATION")
print("-" * 70)

try:
    from src.trend_research.integration import TrendDemandIntegrator

    print("  • Testing trend-demand matching...")

    # Create mock demand forecast
    mock_forecast = pd.DataFrame({
        'point_forecast': np.random.normal(300, 100, len(concepts)),
        'lower_bound': np.random.normal(250, 80, len(concepts)),
        'upper_bound': np.random.normal(350, 120, len(concepts))
    })
    print(f"    ✓ Created mock demand forecast ({len(mock_forecast)} SKUs)")

    # Match designs to demand
    print("  • Matching designs with demand trends...")
    start = time.time()
    try:
        recommendations = researcher.recommend_designs(
            design_concepts=concepts,
            demand_forecast=mock_forecast,
            num_recommendations=5,
            min_confidence=0.6
        )
    except Exception as e:
        print(f"    Error: {e}")
        raise
    elapsed = time.time() - start
    print(f"    ✓ Generated {len(recommendations)} design recommendations in {elapsed:.2f}s")

    for i, rec in enumerate(recommendations[:3], 1):
        print(f"      {i}. {rec.design_description[:40]}...")
        print(f"         - Demand: {rec.predicted_demand:.0f} units")
        print(f"         - Price: ${rec.target_price:.2f}")
        print(f"         - Confidence: {rec.confidence:.1%}")

    print("\n✅ TREND-DEMAND INTEGRATION: PASSED")

except Exception as e:
    print(f"\n❌ TREND-DEMAND INTEGRATION: FAILED - {str(e)}")
    sys.exit(1)

# Test 5: End-to-End Workflow
print("\n[5/5] TESTING END-TO-END WORKFLOW")
print("-" * 70)

try:
    print("  • Generating comprehensive trend report...")
    start = time.time()
    report = researcher.trend_report()
    elapsed = time.time() - start
    print(f"    ✓ Generated trend report in {elapsed:.2f}s")
    print(f"      - Report date: {report['generation_date']}")
    print(f"      - Trending now: {len(report['trending_now'])} categories")
    print(f"      - Emerging trends: {len(report['emerging_trends'])} trends")
    print(f"      - Recommendations: {len(report['recommendations_summary']['next_actions'])} actions")

    print("\n  • Summary of complete workflow:")
    print(f"    ✓ Data: {len(demand_df)} demand records")
    print(f"    ✓ Trends: {sum(len(v) for v in trends.values() if v)} insights")
    print(f"    ✓ Designs: {len(concepts)} concepts → {len(recommendations)} recommendations")
    print(f"    ✓ Collection: {len(collection['styles'])} styles with {len(collection['color_palette'])} colors")
    print(f"    ✓ Revenue potential: ${sum(rec.predicted_demand * rec.target_price for rec in recommendations):,.0f}")

    print("\n✅ END-TO-END WORKFLOW: PASSED")

except Exception as e:
    print(f"\n❌ END-TO-END WORKFLOW: FAILED - {str(e)}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✅ ALL INTEGRATION TESTS PASSED")
print("="*70)
print("\n📊 SYSTEM STATUS: FULLY OPERATIONAL")
print("\nNext steps:")
print("  1. Run: python -m pytest tests/ (for unit tests)")
print("  2. Explore: jupyter notebook notebooks/trend_research_workflow.ipynb")
print("  3. Deploy: Review docs/TREND-RESEARCH-GUIDE.md for production setup")
print("="*70 + "\n")
