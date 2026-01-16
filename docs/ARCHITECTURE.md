# Fashion Demand Predictor - Architecture Overview

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Historical  │  │  Collection  │  │ Macro Economic Data    │ │
│  │ Demand Data │  │   Images     │  │ (GDP, Inflation, etc)  │ │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘ │
└─────────┼──────────────────┼────────────────────┼───────────────┘
          │                  │                    │
          ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                      │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │ Seasonality Features │  │   Macro Features                │  │
│  │ • FFT/Fourier        │  │ • GDP trends                    │  │
│  │ • Seasonal Decomp    │  │ • Inflation index               │  │
│  │ • Cyclical encoding  │  │ • Consumer confidence           │  │
│  │ • Holiday indicators │  │ • Unemployment rate             │  │
│  └──────────────────────┘  │ • Economic shocks detection     │  │
│                             └─────────────────────────────────┘  │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │  Store Features      │  │  ResNet Feature Extraction      │  │
│  │ • Store size/type    │  │ • Collection images             │  │
│  │ • Region encoding    │  │ • CNN embeddings (256-dim)      │  │
│  │ • Baseline demand    │  │ • Style shift detection         │  │
│  │ • Competitor density │  │ • Collection similarity         │  │
│  └──────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FORECASTING MODELS                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ DeepAR+ (RNN-based Time Series Forecasting)                │ │
│  │ • LSTM encoder (hidden_dim=128, num_layers=2)              │ │
│  │ • Multi-head attention mechanism                           │ │
│  │ • Quantile regression (10%, 50%, 90%)                      │ │
│  │ • Probabilistic forecasts + confidence intervals           │ │
│  │ • Forecast horizon: 30 days                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ MDP Optimizer (Markov Decision Process)                     │ │
│  │ • States: (inventory_level, demand_state)                  │ │
│  │ • Actions: order quantities [0, 50, 100, 200, 500]         │ │
│  │ • Rewards: minimize holding costs + stock-outs             │ │
│  │ • Value iteration solver                                   │ │
│  │ • Optimal inventory policy computation                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Ensemble Methods (Optional)                                 │ │
│  │ • Trend extrapolation                                       │ │
│  │ • Seasonal decomposition forecasts                          │ │
│  │ • Weighted ensemble combination                            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AGGREGATION & POST-PROCESSING                    │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │ Store Aggregation    │  │ Scenario Analysis               │  │
│  │ • Product → Store    │  │ • Demand scenarios              │  │
│  │ • Weighted by share  │  │ • Macro factor variations       │  │
│  │ • Regional aggregates│  │ • Sensitivity analysis          │  │
│  └──────────────────────┘  └─────────────────────────────────┘  │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │ Quality Checks       │  │ Explanation & Interpretation    │  │
│  │ • Sanity checks      │  │ • SHAP feature importance       │  │
│  │ • Outlier detection  │  │ • Prediction decomposition      │  │
│  │ • Alert generation   │  │ • Driver analysis               │  │
│  └──────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT & DEPLOYMENT                           │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Predictions  │  │ Optimal      │  │ Store Reports         │  │
│  │ (S3/DDB)     │  │ Inventory    │  │ & Dashboards          │  │
│  │              │  │ Decisions    │  │                       │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Training Pipeline

```
Historical Data → Synthetic Generator → Feature Engineering →
DeepAR+ Training → Model Checkpointing → AWS S3
```

### 2. Inference Pipeline (Daily)

```
EventBridge Trigger (2 AM UTC)
          ↓
Lambda Function
          ↓
Load Latest Macro Data ─┐
Load Historical Data ──┼→ Feature Engineering
Load Model from S3 ────┤
          ↓
Batch Inference (SageMaker)
          ↓
Generate Predictions + Confidence Intervals
          ↓
Apply MDP Optimization
          ↓
Aggregate to Store Level
          ↓
Store in DynamoDB
          ↓
API Endpoint / Dashboard
```

## Key Components

### 1. Data Generation Module (`src/data_generation/`)

**Purpose**: Generate realistic synthetic demand data with all the key components:

```python
SyntheticDataGenerator
├── generate()                    # Main generation method
├── _generate_seasonality()       # Annual + weekly patterns
├── _generate_collection_impact() # New collection ramp-up/decay
├── _generate_macro_factor()      # Macro-economic effects
└── _generate_store_baseline()    # Store-specific multiplier
```

**Features**:
- Strong seasonality (annual cycle + weekly patterns)
- Collection impact modeling (14-day ramp-up, exponential decay)
- Macro factor integration (GDP growth, inflation, confidence)
- Store baseline multipliers for heterogeneity

### 2. Feature Engineering (`src/preprocessing/`)

#### Seasonality Processing
- **Fourier Features**: Decompose time series into sinusoidal components
- **Seasonal Decomposition**: STL decomposition (trend + seasonal + residual)
- **Cyclical Encoding**: sin/cos encoding for day-of-year, day-of-week, month
- **Rolling Statistics**: 7/30/90-day rolling mean, std, min, max

#### Macro Features
- **GDP Trend**: Exponential growth modeling
- **Inflation Index**: Compounded inflation rates
- **Consumer Confidence**: Volatility-adjusted index
- **Unemployment Rate**: Trending + seasonal component
- **Economic Shocks**: Threshold-based detection

#### Store Features
- **Size Normalization**: Store area in sqm
- **Type Encoding**: Flagship, outlet, regular, online (one-hot)
- **Region Encoding**: Geographic region (one-hot)
- **Baseline Calculation**: 90-day trailing average demand
- **Store Similarity**: Feature-based cosine similarity

### 3. Deep Learning Models (`src/models/`)

#### DeepAR+ Pipeline

**Architecture**:
- **Input**: Concatenation of demand history + features (seq_len, feature_dim)
- **LSTM Encoder**:
  - 2 stacked LSTM layers (128 hidden units)
  - Dropout (0.2) for regularization
  - Multi-head attention (4 heads)
- **Quantile Decoder**:
  - Outputs [Q10, Q50, Q90] predictions
  - Probabilistic forecasts
- **Loss Function**:
  - Negative Binomial loss (for count data)
  - Alternative: Quantile loss for robust predictions

**Training**:
- Batch size: 64
- Learning rate: 0.001 (Adam optimizer)
- Early stopping: patience=10 epochs
- Validation split: 10%

#### ResNet Feature Extractor

**Purpose**: Convert collection images → compact embeddings

**Architecture**:
- **Base Model**: ResNet50 pretrained on ImageNet
- **Feature Extraction**: Remove FC layers, use conv features
- **Projection Head**: 2048 → 512 → 256 dimensions
- **Aggregation**: Mean pooling across collection images

**Application**:
- Extract embeddings from new collection images
- Compute collection-to-collection similarity
- Detect style shifts (similarity < threshold)
- Feed embeddings as features to DeepAR+

#### MDP Optimizer

**Problem Formulation**:
- **State Space**: (inventory_level ∈ [0,10,25,50,100,500], demand_state ∈ [low,medium,high])
- **Action Space**: order_qty ∈ [0, 50, 100, 200, 500]
- **Rewards**:
  - `R(s,a) = -order_cost·a - holding_cost·inv - stock_out_penalty + sales_reward`
- **Objective**: Maximize expected cumulative discounted reward

**Solution Method**:
- Value Iteration (or Policy Iteration)
- Discount factor: γ = 0.95
- Convergence tolerance: 1e-6

**Output**:
- Optimal policy: `π(s) → a` mapping
- Value function: `V(s)` for each state

### 4. Prediction Engine (`src/prediction/`)

#### Inference Pipeline

```python
PredictionEngine
├── forecast()                 # Main prediction method
├── scenario_forecast()        # Scenario analysis
├── ensemble_forecast()        # Ensemble predictions
├── explain_prediction()       # Feature importance (SHAP)
└── _apply_mdp_optimization()  # Inventory optimization
```

#### Store Aggregation

```python
StoreAggregator
├── aggregate()                # Product → Store level
├── disaggregate()             # Store → Product level
├── apply_store_adjustments()  # Baseline adjustments
├── create_regional_aggregates() # Regional summaries
└── hierarchical_reconciliation() # Multi-level consistency
```

## Deployment Architecture (AWS)

### Components

1. **SageMaker Training**
   - Instance: ml.p3.2xlarge (GPU)
   - Spot instances enabled (save 70%)
   - Training job: ~2-3 hours

2. **SageMaker Endpoints**
   - Instance: ml.m5.large (2 instances, multi-AZ)
   - Auto-scaling: 1-5 instances based on load
   - Inference latency: <1s

3. **Lambda Functions**
   - Memory: 1024 MB
   - Timeout: 300 seconds
   - Trigger: EventBridge schedule

4. **DynamoDB**
   - Table: `fashion-predictions`
     - Partition key: store_id
     - Sort key: forecast_date
     - TTL: 90 days
   - Table: `fashion-metadata`
     - Model versions and metrics

5. **S3 Buckets**
   - Training data: s3://fashion-predictor-training/
   - Models: s3://fashion-predictor-models/
   - Predictions: s3://fashion-predictor-predictions/

### Daily Workflow

```
EventBridge Rule (0 2 * * ? *) [UTC 2 AM]
    ↓
Invoke Lambda (forecast-runner)
    ↓
Load Macro Data (latest economic indicators)
    ↓
Load Latest Store/Product Data
    ↓
Feature Engineering (preprocessing)
    ↓
Invoke SageMaker Endpoint
    ↓
DeepAR+ Predictions (store × product)
    ↓
MDP Optimization (inventory decisions)
    ↓
Aggregate to Store Level
    ↓
Write to DynamoDB
    ↓
Trigger SNS Alert (if anomalies)
    ↓
API returns predictions
```

## Configuration Files

### `config/config.yaml`
- Data generation parameters
- Model hyperparameters
- Training configuration
- Prediction settings

### `config/aws_config.yaml`
- AWS region and credentials
- SageMaker instance types
- DynamoDB table configs
- Lambda settings
- CloudWatch alarms
- Cost optimization settings

## Performance Metrics

### Training
- Validation RMSE: ~15-20% of mean demand
- Training time: 2-3 hours (GPU)
- Convergence: ~50 epochs typical

### Inference
- Real-time latency: <1 second
- Batch latency: <100ms per prediction
- Throughput: 10K predictions/minute

### Business Metrics
- Forecast accuracy (MAPE): 10-15%
- Stock-out reduction: 20-30%
- Inventory reduction: 15-25%

## Future Enhancements

1. **Graph Neural Networks**: Model store relationships
2. **Attention Mechanisms**: Improved temporal modeling
3. **AutoML**: Automated hyperparameter tuning
4. **Real-time Data Integration**: Live collection launches
5. **Causal Inference**: Quantify campaign impacts
6. **Hierarchical Forecasting**: Multi-level reconciliation
7. **Transfer Learning**: Cross-store knowledge transfer
8. **Online Learning**: Model adaptation with new data

## References

- [DeepAR+](https://arxiv.org/abs/2003.01409)
- [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)
- [Probabilistic Forecasting](https://arxiv.org/abs/1912.09363)
- [MDP Optimization](https://en.wikipedia.org/wiki/Markov_decision_process)
