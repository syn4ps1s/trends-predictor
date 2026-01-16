# AWS Setup Guide - Fashion Demand Predictor

Guía completa para desplegar el predictor de demanda de moda en AWS usando SageMaker, Lambda, DynamoDB y S3.

## Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Configuración Inicial](#configuración-inicial)
3. [SageMaker Setup](#sagemaker-setup)
4. [Deployment del Modelo](#deployment-del-modelo)
5. [Serverless Inference](#serverless-inference)
6. [Monitoreo y Alertas](#monitoreo-y-alertas)
7. [Costos y Optimización](#costos-y-optimización)

## Requisitos Previos

- Cuenta de AWS activa
- CLI de AWS configurada: `aws configure`
- Rol IAM con permisos SageMaker, S3, Lambda, DynamoDB, ECR

### Permisos IAM Necesarios

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:*",
        "s3:*",
        "ecr:*",
        "lambda:*",
        "dynamodb:*",
        "logs:*",
        "cloudwatch:*"
      ],
      "Resource": "*"
    }
  ]
}
```

## Configuración Inicial

### 1. Crear S3 Buckets

```bash
# Bucket para datos de entrenamiento
aws s3 mb s3://fashion-predictor-training --region us-east-1

# Bucket para modelos
aws s3 mb s3://fashion-predictor-models --region us-east-1

# Bucket para predicciones
aws s3 mb s3://fashion-predictor-predictions --region us-east-1

# Habilitar versionado
aws s3api put-bucket-versioning \
  --bucket fashion-predictor-models \
  --versioning-configuration Status=Enabled
```

### 2. Crear DynamoDB Tables

```bash
# Tabla de predicciones
aws dynamodb create-table \
  --table-name fashion-predictions \
  --attribute-definitions \
    AttributeName=store_id,AttributeType=N \
    AttributeName=forecast_date,AttributeType=S \
  --key-schema \
    AttributeName=store_id,KeyType=HASH \
    AttributeName=forecast_date,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Tabla de metadatos
aws dynamodb create-table \
  --table-name fashion-metadata \
  --attribute-definitions \
    AttributeName=model_id,AttributeType=S \
    AttributeName=version,AttributeType=N \
  --key-schema \
    AttributeName=model_id,KeyType=HASH \
    AttributeName=version,KeyType=RANGE \
  --billing-mode PROVISIONED \
  --provisioned-throughput ReadCapacityUnits=10,WriteCapacityUnits=10 \
  --region us-east-1
```

### 3. Crear IAM Role para SageMaker

```bash
# Crear rol
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": "sagemaker.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# Adjuntar política
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

## SageMaker Setup

### 1. Crear SageMaker Domain

```bash
aws sagemaker create-domain \
  --domain-name fashion-predictor-domain \
  --auth-mode IAM \
  --default-user-settings \
    ExecutionRole=arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole
```

### 2. Preparar Código de Entrenamiento

Crear archivo `train.py` en `aws/training/`:

```python
import argparse
import json
import numpy as np
import pandas as pd
from src.models import DeepARPipeline
from src.data_generation import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()

    # Generar datos
    generator = SyntheticDataGenerator(num_stores=100, num_products=500)
    data = generator.generate()

    # Entrenar modelo
    pipeline = DeepARPipeline()
    pipeline.build_model(input_dim=10)
    pipeline.train(data, epochs=args.epochs)

    # Guardar modelo
    import torch
    torch.save(pipeline.model.state_dict(), '/opt/ml/model/model.pth')

if __name__ == '__main__':
    main()
```

### 3. Crear Imagen Docker

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /src/
COPY aws/training/train.py /train.py

ENTRYPOINT ["python", "/train.py"]
```

Build y push:

```bash
# Login a ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -t fashion-predictor-training:latest .

# Tag
docker tag fashion-predictor-training:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fashion-predictor-training:latest

# Push
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fashion-predictor-training:latest
```

### 4. Lanzar Training Job

```python
import sagemaker
from sagemaker.estimator import Estimator

role_arn = "arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole"
image_uri = "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fashion-predictor-training:latest"

estimator = Estimator(
    image_uri=image_uri,
    role=role_arn,
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    output_path="s3://fashion-predictor-models/training-output",
)

estimator.fit("s3://fashion-predictor-training/data/")
```

## Deployment del Modelo

### 1. SageMaker Endpoint (Real-time)

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://fashion-predictor-models/model.tar.gz",
    role=role_arn,
    framework_version="2.0",
    py_version="py310",
    entry_point="inference.py",
    source_dir=".",
)

predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.m5.large",
    endpoint_name="fashion-predictor-endpoint-prod",
)
```

### 2. Batch Transform (Batch Processing)

```python
from sagemaker.transformer import Transformer

transformer = Transformer(
    model_name="fashion-predictor-model",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://fashion-predictor-predictions/batch/",
)

transformer.transform(
    data="s3://fashion-predictor-training/input/",
    content_type="text/csv",
    split_type="Line",
)
```

## Serverless Inference

### 1. Lambda Function para Inference

```python
import json
import boto3
import numpy as np

sagemaker_runtime = boto3.client('sagemaker-runtime')
dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    store_id = event.get('store_id')
    product_id = event.get('product_id')

    # Preparar input
    input_data = prepare_input(store_id, product_id)

    # Llamar endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='fashion-predictor-endpoint-prod',
        ContentType='application/json',
        Body=json.dumps(input_data)
    )

    # Parsear respuesta
    prediction = json.loads(response['Body'].read().decode())

    # Guardar en DynamoDB
    table = dynamodb.Table('fashion-predictions')
    table.put_item(Item={
        'store_id': store_id,
        'product_id': product_id,
        'forecast': float(prediction['forecast'][0]),
        'confidence_interval': {
            'lower': float(prediction['lower'][0]),
            'upper': float(prediction['upper'][0])
        }
    })

    return {
        'statusCode': 200,
        'body': json.dumps(prediction)
    }
```

Deploy:

```bash
zip -r lambda_function.zip lambda_handler.py

aws lambda create-function \
  --function-name fashion-predictor-inference \
  --runtime python3.11 \
  --role arn:aws:iam::ACCOUNT_ID:role/LambdaExecutionRole \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://lambda_function.zip
```

### 2. EventBridge para Ejecución Programada

```bash
# Regla para ejecutar cada día a las 2 AM UTC
aws events put-rule \
  --name daily-forecast-rule \
  --schedule-expression "cron(0 2 * * ? *)" \
  --state ENABLED

# Target: Lambda function
aws events put-targets \
  --rule daily-forecast-rule \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:ACCOUNT_ID:function:fashion-predictor-inference"
```

## Monitoreo y Alertas

### 1. CloudWatch Metrics

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Registrar métrica personalizada
cloudwatch.put_metric_data(
    Namespace='FashionPredictor',
    MetricData=[
        {
            'MetricName': 'ForecastRMSE',
            'Value': rmse_value,
            'Unit': 'None'
        },
        {
            'MetricName': 'InferenceLatency',
            'Value': latency_ms,
            'Unit': 'Milliseconds'
        }
    ]
)
```

### 2. Crear Alarmas

```bash
# Alarma para accuracy
aws cloudwatch put-metric-alarm \
  --alarm-name high-forecast-error \
  --alarm-description "Alert if forecast RMSE is high" \
  --metric-name ForecastRMSE \
  --namespace FashionPredictor \
  --statistic Average \
  --period 300 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:alerts

# Alarma para latencia
aws cloudwatch put-metric-alarm \
  --alarm-name high-inference-latency \
  --metric-name InferenceLatency \
  --namespace FashionPredictor \
  --statistic Average \
  --period 60 \
  --threshold 1000 \
  --comparison-operator GreaterThanThreshold
```

### 3. Dashboard

```python
cloudwatch.put_dashboard(
    DashboardName='fashion-predictor-dashboard',
    DashboardBody=json.dumps({
        'widgets': [
            {
                'type': 'metric',
                'properties': {
                    'metrics': [
                        ['FashionPredictor', 'ForecastRMSE'],
                        ['FashionPredictor', 'InferenceLatency']
                    ],
                    'period': 300,
                    'stat': 'Average',
                    'region': 'us-east-1'
                }
            }
        ]
    })
)
```

## Costos y Optimización

### Estimación de Costos Mensuales

```
SageMaker Training (ml.p3.2xlarge): ~$4,000/mes
SageMaker Endpoint (ml.m5.large x2): ~$600/mes
Lambda invocations (1M/mes): ~$20/mes
DynamoDB on-demand: ~$50/mes
S3 storage & transfer: ~$100/mes
CloudWatch logs: ~$50/mes
─────────────────
TOTAL: ~$4,820/mes
```

### Optimizaciones

1. **Spot Instances**: Reducir costos de entrenamiento hasta 70%
2. **Reserved Capacity**: Para endpoints permanentes
3. **Batch Processing**: En lugar de real-time para predicciones no urgentes
4. **Lambda Provisioned Concurrency**: Optimizar cold starts

### Configurar Spot Training

```python
estimator = Estimator(
    image_uri=image_uri,
    role=role_arn,
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    use_spot_instances=True,
    max_wait_time_in_seconds=3600,  # Max espera
    output_path="s3://fashion-predictor-models/training-output",
)
```

## Troubleshooting

### Logs de Training

```bash
aws logs tail /aws/sagemaker/fashion-predictor --follow
```

### Revisar Estado del Endpoint

```bash
aws sagemaker describe-endpoint --endpoint-name fashion-predictor-endpoint-prod
```

### Inspeccionar S3

```bash
aws s3 ls s3://fashion-predictor-models/ --recursive
```

## Contacto y Soporte

Para issues o preguntas, crear issue en el repositorio.
