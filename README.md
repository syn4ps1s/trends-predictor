# Fashion Demand Predictor 👗📊

Predictor avanzado de demanda de moda, textil y retail con estacionalidad fuerte, nuevas colecciones y análisis macroeconómico.

## Características Principales

- **Estacionalidad Fuerte**: Modela ciclos estacionales en demanda de moda
- **Nuevas Colecciones**: Impacto de lanzamientos de colecciones en demanda
- **Análisis Macroeconómico**: Integra premisas macro (economía, tendencias globales)
- **Granularidad por Tienda**: Predicciones desagregadas por punto de venta
- **Datos Sintéticos**: Generador de datos para entrenamiento sin exponer datos reales

## Stack Tecnológico

### ML/AI
- **DeepAR+**: Modelo de forecasting basado en RNN (Amazon SageMaker)
- **ResNet**: Feature extractor para imágenes de colecciones → embeddings
- **MDP (Markov Decision Process)**: Motor de decisiones para optimización de inventario
- **Surrogate Models**: Modelos rápidos para simulación y optimización

### Cloud (AWS)
- **Amazon SageMaker**: Entrenamiento y deployment de modelos
- **S3**: Storage de datos
- **Lambda**: Inference serverless
- **DynamoDB**: Almacenamiento de predicciones y metadatos
- **EventBridge**: Orquestación de pipelines

### Data & Computing
- **Python 3.10+**: Principal lenguaje
- **PyTorch**: Framework para DeepAR+ y ResNet
- **GluonTS**: Librería especializada en forecasting
- **Pandas/NumPy**: Manipulación de datos
- **AWS SDK (Boto3)**: Integración AWS

## Estructura del Proyecto

```
trends-predictor/
├── data/                          # Datos de entrada y salida
│   ├── raw/                       # Datos crudos
│   ├── synthetic/                 # Datos generados
│   └── processed/                 # Datos procesados
├── src/
│   ├── data_generation/           # Generador de datos sintéticos
│   │   ├── __init__.py
│   │   ├── synthetic_generator.py
│   │   └── collection_simulator.py
│   ├── preprocessing/             # Feature engineering
│   │   ├── __init__.py
│   │   ├── seasonality.py
│   │   ├── macro_features.py
│   │   └── store_features.py
│   ├── models/                    # Modelos ML
│   │   ├── __init__.py
│   │   ├── deepar_pipeline.py
│   │   ├── resnet_extractor.py
│   │   ├── mdp_optimizer.py
│   │   └── surrogate.py
│   └── prediction/                # Pipeline de predicción
│       ├── __init__.py
│       ├── inference.py
│       ├── store_aggregation.py
│       └── aws_sagemaker.py
├── notebooks/                     # Análisis exploratorio
├── config/                        # Configuraciones
│   ├── config.yaml
│   └── aws_config.yaml
├── aws/                           # Scripts/templates AWS
│   ├── sagemaker_training.py
│   ├── lambda_functions/
│   └── cloudformation/
├── tests/                         # Tests unitarios
├── docs/                          # Documentación
└── requirements.txt
```

## Flujo de Datos

```
Datos Crudos / Sintéticos
    ↓
Preprocesamiento
    ├─→ Estacionalidad (FFT, Seasonal Decomposition)
    ├─→ Features Macro (indicadores económicos)
    ├─→ Features de Tienda (tamaño, región, tipo)
    └─→ Imágenes Colecciones → ResNet → Embeddings
    ↓
DeepAR+ Training (SageMaker)
    ↓
Predicción por Producto/Tienda
    ↓
MDP Optimizer (Decisiones de Inventario)
    ↓
Predicciones Finales → DynamoDB / S3
```

## Instalación y Setup

```bash
# Clonar y configurar
git clone <repo>
cd trends-predictor
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar AWS (si no está)
aws configure

# Configurar variables de entorno
cp .env.example .env
```

## Uso Rápido

```python
from src.data_generation import SyntheticDataGenerator
from src.models import DeepARPipeline

# 1. Generar datos sintéticos
generator = SyntheticDataGenerator(num_stores=100, num_products=500)
synthetic_data = generator.generate()

# 2. Entrenar DeepAR+
pipeline = DeepARPipeline(config='config/config.yaml')
pipeline.train(synthetic_data)

# 3. Predecir
predictions = pipeline.predict(
    forecast_horizon=30,
    store_ids=[1, 2, 3]
)
```

## Configuración AWS

Ver `docs/aws-setup.md` para:
- Setup de SageMaker
- Políticas IAM necesarias
- Deployment de modelos
- Monitoreo en producción

## Próximos Pasos

- [ ] Integración con datos reales
- [ ] Dashboard de monitoreo
- [ ] API REST en Lambda
- [ ] A/B testing framework
- [ ] Optimización de costos AWS

## Contacto y Contribuciones

Para reportes de bugs o contribuciones, abrir issue en el repositorio.
