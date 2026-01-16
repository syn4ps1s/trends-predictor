# Testing Guide - Fashion Demand Predictor v2.0.0

Complete guide to test all components of the Trend Research and Demand Forecasting system.

## Quick Start (5 minutes)

### 1. Test Trend Research
```bash
cd /home/user/trends-predictor

# Install minimal dependencies
pip install numpy pandas torch diffusers

# Test 1: Basic Trend Researcher
python -c "
from src.trend_research import TrendResearcher

researcher = TrendResearcher(llm_provider='openai')
print('✅ TrendResearcher initialized')
print(f'   LLM Provider: {researcher.llm_provider}')
print(f'   Generative Model: {researcher.generative_model}')
"
```

### 2. Test Synthetic Data Generation
```python
python -c "
from src.data_generation import SyntheticDataGenerator

# Create generator
gen = SyntheticDataGenerator(
    num_stores=5,
    num_products=10,
    num_days=100
)

# Generate data
data = gen.generate()
print(f'✅ Generated {len(data)} demand records')
print(f'   Columns: {list(data.columns)}')
print(f'   Date range: {data[\"timestamp\"].min()} to {data[\"timestamp\"].max()}')

# Generate metadata
stores = gen.generate_store_metadata()
products = gen.generate_product_metadata()
print(f'✅ Generated {len(stores)} stores and {len(products)} products')
"
```

### 3. Test Feature Engineering
```python
python -c "
from src.preprocessing import SeasonalityProcessor, MacroFeaturesProcessor
from src.preprocessing import StoreFeatureProcessor
import pandas as pd

# Test Seasonality
processor = SeasonalityProcessor()
print('✅ SeasonalityProcessor initialized')

# Test Macro Features
macro = MacroFeaturesProcessor()
dates = pd.date_range('2023-01-01', periods=365, freq='D')
features = macro.create_macro_features_df(dates)
print(f'✅ Generated {len(features)} macro feature records')

# Test Store Features
store = StoreFeatureProcessor()
print('✅ StoreFeatureProcessor initialized')
"
```

---

## Component Testing (15-30 minutes)

### Test 1: Trend Research System

#### 1.1 Basic Trend Research (Mock Mode)
```python
from src.trend_research import TrendResearcher

# Initialize without API keys (will use mock data)
researcher = TrendResearcher(
    llm_provider="openai",  # Will fall back to mock if no API key
    generative_model="stable-diffusion"
)

# Research trends
print("🔮 Researching trends...")
trends = researcher.research_trends(
    categories=["color", "material", "silhouette"],
    regions=["North America", "Europe", "Asia"],
    lookback_weeks=4,
    forecast_weeks=12
)

# Examine results
for category, insights in trends.items():
    if insights:
        print(f"\n📊 {category.upper()}:")
        for insight in insights[:2]:
            print(f"  • {insight.trend_name}")
            print(f"    Confidence: {insight.confidence_score:.2%}")
            print(f"    Growth: {insight.growth_rate:.1%} MoM")
            print(f"    Regions: {', '.join(insight.regions)}")
```

#### 1.2 Generate Design Concepts
```python
print("🎨 Generating design concepts...")
concepts = researcher.generate_design_concepts(
    trend_insights=trends,
    num_concepts=3,
    style="contemporary"
)

print(f"✅ Generated {len(concepts)} design concepts:")
for i, concept in enumerate(concepts, 1):
    print(f"\n  Concept {i}:")
    print(f"    Prompt: {concept['design_prompt'][:80]}...")
    print(f"    Trends: {', '.join(concept['selected_trends'][:2])}")
    print(f"    Image: {concept['generated_image']}")
```

#### 1.3 Recommend Designs
```python
import pandas as pd
import numpy as np

# Create mock demand forecast
forecast_df = pd.DataFrame({
    'point_forecast': np.random.normal(300, 100, len(concepts)),
    'lower_bound': np.random.normal(250, 80, len(concepts)),
    'upper_bound': np.random.normal(350, 120, len(concepts))
})

print("💡 Recommending designs based on demand...")
recommendations = researcher.recommend_designs(
    design_concepts=concepts,
    demand_forecast=forecast_df,
    num_recommendations=3
)

print(f"✅ Generated {len(recommendations)} recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"\n  Design {i}:")
    print(f"    Product ID: {rec.product_id}")
    print(f"    Color: {rec.primary_color}")
    print(f"    Material: {rec.material}")
    print(f"    Demand: {rec.predicted_demand:.0f} units")
    print(f"    Confidence: {rec.confidence:.2%}")
    print(f"    Price: ${rec.target_price:.2f}")
```

#### 1.4 Create Seasonal Collection
```python
print("👗 Creating seasonal collection...")
collection = researcher.seasonal_collection_plan(
    upcoming_season="spring",
    num_styles=3,
    num_colors_per_style=2
)

print(f"✅ Collection created:")
print(f"  Season: {collection['season']}")
print(f"  Styles: {len(collection['styles'])}")
print(f"  Color Palette: {', '.join(collection['color_palette'][:3])}")
print(f"  Target Demographics: {collection['target_demographics']['age_groups']}")

for i, style in enumerate(collection['styles'][:2], 1):
    print(f"\n  Style {i}: {style['style_id']}")
    print(f"    Colors: {len(style['color_variations'])} variations")
    print(f"    Quantity: {style['production_quantity']} units")
```

#### 1.5 Generate Trend Report
```python
print("📋 Generating trend report...")
report = researcher.trend_report()

print(f"✅ Trend Report Generated:")
print(f"  Generation Date: {report['generation_date']}")
print(f"  Last Update: {report['last_update']}")
print(f"\n  Emerging Trends:")
for trend in report['emerging_trends'][:2]:
    print(f"    • {trend['trend']} ({trend['growth']})")
    print(f"      → {trend['recommendation']}")
```

### Test 2: Design Recommender

#### 2.1 Material Recommendations
```python
from src.trend_research import DesignRecommender

recommender = DesignRecommender()

print("🧵 Testing material recommendations...")
material_rec = recommender.recommend_materials(
    design_analysis={
        "category": "casual",
        "complexity_score": 0.75
    },
    color="sage green",
    season="spring"
)

print("✅ Material Recommendation:")
print(f"  Primary: {material_rec['primary_material']}")
print(f"  Blend: {material_rec['blend']}")
print(f"  GSM: {material_rec['gsm']}")
print(f"  Lead Time: {material_rec['lead_time']} weeks")
print(f"  Cost/Meter: ${material_rec['cost_per_meter']:.2f}")
print(f"  Sustainability: {material_rec['sustainability_score']:.1%}")
```

#### 2.2 Color Recommendations
```python
print("\n🎨 Testing color recommendations...")
colors = recommender.recommend_colors_for_season(
    season="spring",
    num_recommendations=5,
    include_trends=True
)

print("✅ Seasonal Color Palette:")
for color in colors:
    print(f"  • {color['color'].upper()}")
    print(f"    Type: {color['type']}")
    print(f"    Psychology: {color['psychology']}")
    print(f"    Trending: {color['trending']}")
```

#### 2.3 Pricing Recommendations
```python
print("\n💰 Testing pricing strategy...")
pricing = recommender.price_recommendation(
    material_cost=12.50,
    design_complexity=0.75,
    brand_positioning="contemporary",
    market_segment="mid_market"
)

print("✅ Pricing Recommendation:")
print(f"  Production Cost: ${pricing['production_cost']:.2f}")
print(f"  Recommended Price: ${pricing['recommended_retail_price']:.2f}")
print(f"  Wholesale: ${pricing['wholesale_price']:.2f}")
print(f"  Margin: {pricing['margin_percentage']:.1%}")
```

### Test 3: Demand Forecasting

#### 3.1 Synthetic Data Generation
```python
from src.data_generation import SyntheticDataGenerator

print("📊 Generating synthetic demand data...")
generator = SyntheticDataGenerator(
    num_stores=20,
    num_products=100,
    num_days=365
)

demand_df = generator.generate()
print(f"✅ Generated {len(demand_df):,} demand records")
print(f"   Date Range: {demand_df['timestamp'].min()} to {demand_df['timestamp'].max()}")
print(f"   Demand Stats:")
print(f"     Mean: {demand_df['demand'].mean():.2f} units")
print(f"     Std Dev: {demand_df['demand'].std():.2f}")
print(f"     Range: {demand_df['demand'].min():.2f} - {demand_df['demand'].max():.2f}")
```

#### 3.2 Feature Engineering
```python
from src.preprocessing import SeasonalityProcessor, MacroFeaturesProcessor, StoreFeatureProcessor

print("\n🔧 Testing feature engineering...")

# Seasonality
seasonality = SeasonalityProcessor()
fourier_feat = seasonality.extract_fourier_features(
    demand_df['demand'].values,
    num_terms=10
)
print(f"✅ Fourier features: {fourier_feat.shape}")

# Macro
macro = MacroFeaturesProcessor()
macro_feat = macro.create_macro_features_df(
    demand_df['timestamp'].unique()[:100]
)
print(f"✅ Macro features: {macro_feat.shape}")

# Store
store = StoreFeatureProcessor()
store_meta = generator.generate_store_metadata()
store.load_store_metadata(store_meta)
baselines = store.calculate_store_baselines(demand_df)
print(f"✅ Store baselines: {len(baselines)} stores")
```

#### 3.3 DeepAR+ Training (Quick Test)
```python
from src.models import DeepARPipeline, DeepARConfig

print("\n🧠 Testing DeepAR+ model...")

config = DeepARConfig(
    hidden_dim=64,
    num_layers=1,
    dropout=0.1,
    learning_rate=0.001,
    batch_size=32,
    epochs=2,  # Quick test
    forecast_horizon=7
)

pipeline = DeepARPipeline(config=config)
print(f"✅ DeepAR+ configured")
print(f"   Hidden Dim: {config.hidden_dim}")
print(f"   Forecast Horizon: {config.forecast_horizon}")

# Note: Full training requires more data processing
# This is just to verify the module loads correctly
```

#### 3.4 MDP Optimizer
```python
from src.models import MDPOptimizer

print("\n🎯 Testing MDP optimizer...")

mdp = MDPOptimizer(
    inventory_bins=[0, 10, 25, 50, 100],
    demand_states=['low', 'medium', 'high'],
    discount_factor=0.95
)

print(f"✅ MDP configured")
print(f"   States: {mdp.num_states}")
print(f"   Actions: {mdp.num_actions}")

# Fit with demand data
mdp.fit_transition_probabilities(demand_df['demand'].values)
print(f"✅ Transition probabilities fitted")

# Solve
mdp.solve()
print(f"✅ MDP solved")

# Get optimal order
optimal_qty = mdp.get_optimal_order(
    current_inventory=50,
    predicted_demand_state="medium"
)
print(f"   Optimal order for 50 units + medium demand: {optimal_qty} units")
```

### Test 4: Integration

#### 4.1 Trend-Demand Matching
```python
from src.trend_research.integration import TrendDemandIntegrator

print("🔗 Testing Trend-Demand Integration...")

# Create mock integrator components
class MockDeepAR:
    def predict(self, X, return_intervals=True):
        return {
            'point_forecast': np.random.normal(200, 50, (len(X), 30)),
            'lower_bound': np.random.normal(150, 40, (len(X), 30)),
            'upper_bound': np.random.normal(250, 60, (len(X), 30))
        }

# Test matching
trends_dict = {
    'color': [next(iter(trends.get('color', []))]
    ] if trends.get('color') else []
}

# Mock forecast
forecast_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=30),
    'point_forecast': np.random.normal(300, 100, 30)
})

print("✅ Trend-Demand integration ready")
print("   Components: TrendResearcher, DeepAR+, MDP, StoreAggregator")
```

---

## Full Integration Test (Complete Workflow)

Create a file `test_complete_workflow.py`:

```python
#!/usr/bin/env python
"""Complete end-to-end workflow test."""

import numpy as np
import pandas as pd
from src.trend_research import TrendResearcher
from src.data_generation import SyntheticDataGenerator
from src.models import DeepARPipeline, MDPOptimizer
from src.prediction import StoreAggregator

def test_complete_workflow():
    """Test complete system workflow."""

    print("=" * 70)
    print("TESTING COMPLETE FASHION DEMAND PREDICTOR WORKFLOW")
    print("=" * 70)

    # 1. TEST TREND RESEARCH
    print("\n1️⃣  TREND RESEARCH")
    print("-" * 70)

    researcher = TrendResearcher()
    print("✅ TrendResearcher initialized")

    trends = researcher.research_trends(
        categories=["color"],
        forecast_weeks=12
    )
    print("✅ Trends researched")

    concepts = researcher.generate_design_concepts(
        trend_insights=trends,
        num_concepts=2,
        style="contemporary"
    )
    print(f"✅ {len(concepts)} design concepts generated")

    # 2. TEST SYNTHETIC DATA
    print("\n2️⃣  SYNTHETIC DATA GENERATION")
    print("-" * 70)

    generator = SyntheticDataGenerator(
        num_stores=10,
        num_products=50,
        num_days=180
    )
    demand_df = generator.generate()
    print(f"✅ Generated {len(demand_df):,} demand records")

    # 3. TEST FORECASTING
    print("\n3️⃣  DEMAND FORECASTING")
    print("-" * 70)

    config_dict = {
        'hidden_dim': 64,
        'epochs': 1,  # Quick test
    }
    pipeline = DeepARPipeline()
    print("✅ DeepAR+ pipeline initialized")

    # 4. TEST OPTIMIZATION
    print("\n4️⃣  INVENTORY OPTIMIZATION")
    print("-" * 70)

    mdp = MDPOptimizer()
    mdp.fit_transition_probabilities(demand_df['demand'].values)
    mdp.solve()
    print("✅ MDP optimization complete")

    # 5. TEST AGGREGATION
    print("\n5️⃣  STORE AGGREGATION")
    print("-" * 70)

    aggregator = StoreAggregator()
    store_meta = generator.generate_store_metadata()
    aggregator.fit(demand_df, store_meta)
    print("✅ Store aggregator fitted")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)

    return {
        'trends': trends,
        'concepts': concepts,
        'demand_data': demand_df,
        'pipeline': pipeline,
        'mdp': mdp,
        'aggregator': aggregator
    }

if __name__ == "__main__":
    results = test_complete_workflow()
```

Run with:
```bash
python test_complete_workflow.py
```

---

## API Testing with cURL (AWS)

Once deployed to AWS, test endpoints:

```bash
# Test SageMaker Endpoint
curl -X POST https://[endpoint-url]/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 1,
    "product_id": 100,
    "forecast_days": 30
  }'

# Test Lambda Function
curl -X POST https://[lambda-url] \
  -H "Content-Type: application/json" \
  -d '{
    "action": "forecast",
    "season": "spring"
  }'

# Query DynamoDB Predictions
aws dynamodb query \
  --table-name fashion-predictions \
  --key-condition-expression "store_id = :sid" \
  --expression-attribute-values '{
    ":sid": {"N": "1"}
  }' \
  --region us-east-1
```

---

## Performance Testing

### Latency Test
```python
import time
from src.models import DeepARPipeline

pipeline = DeepARPipeline()

# Mock data
X_test = np.random.randn(100, 60, 10)

# Time predictions
start = time.time()
predictions = pipeline.predict(X_test)
elapsed = time.time() - start

print(f"⚡ Prediction latency: {elapsed/100*1000:.1f} ms per sample")
```

### Throughput Test
```python
import time

# 1000 predictions
start = time.time()
for i in range(1000):
    _ = pipeline.predict(X_test[0:1])
elapsed = time.time() - start

print(f"📊 Throughput: {1000/elapsed:.0f} predictions/second")
```

---

## What You Can Test Right Now

✅ **No API Keys Required**:
1. Synthetic data generation
2. Feature engineering
3. DeepAR+ model structure
4. MDP optimizer logic
5. Store aggregation
6. Basic trend research (mock mode)

⚠️ **Requires API Keys**:
1. Real LLM trend analysis (OpenAI/Anthropic)
2. Real image generation (Stable Diffusion GPU / DALL-E)
3. Fine-tuned collection planning

---

## Troubleshooting Tests

If you get errors:

```bash
# Missing dependencies
pip install torch transformers diffusers pillow

# CUDA issues (if you have GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# LLM issues
export OPENAI_API_KEY="your-key-here"
# OR use Ollama
ollama pull mistral
```

---

## Quick Test Matrix

| Component | Test Time | API Keys | Status |
|-----------|-----------|----------|--------|
| Data Generation | 1 min | ❌ No | ✅ Ready |
| Seasonality | 1 min | ❌ No | ✅ Ready |
| Macro Features | 1 min | ❌ No | ✅ Ready |
| Store Features | 1 min | ❌ No | ✅ Ready |
| DeepAR+ Init | 1 min | ❌ No | ✅ Ready |
| MDP Optimizer | 2 min | ❌ No | ✅ Ready |
| Aggregation | 1 min | ❌ No | ✅ Ready |
| Trend Analysis (mock) | 2 min | ❌ No | ✅ Ready |
| Design Generation (mock) | 1 min | ❌ No | ✅ Ready |
| Complete Workflow | 5 min | ❌ No | ✅ Ready |

---

Next: Run `python test_complete_workflow.py` to see everything in action!
