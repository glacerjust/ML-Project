# Production ML Platform — Technical Specification

**Status:** Planning
**Version:** 1.0
**Project Type:** End-to-End ML Engineering / MLOps Portfolio Project
**Primary Goal:** Build a production-oriented ML lifecycle platform, not merely a machine-learning notebook.

---

# 1. Project Definition

## 1.1 Problem

Build a production-oriented machine learning platform for **customer churn prediction**.

The system must support the complete lifecycle of a machine learning model:

```text
Data
  ↓
Validation
  ↓
Feature Engineering
  ↓
Training
  ↓
Experiment Tracking
  ↓
Model Registry
  ↓
Model Evaluation
  ↓
Model Serving
  ↓
Production Monitoring
  ↓
Drift / Performance Detection
  ↓
Retraining
  ↓
Model Evaluation
  ↓
Promote / Reject
```

The central engineering question is:

> **How can we reliably move an ML model from training to production and continuously determine whether the deployed model is still suitable for production?**

The project should demonstrate both:

* **Machine Learning knowledge**
* **ML Engineering / MLOps knowledge**

The model itself is not the final product.

The **ML lifecycle and platform are the primary engineering product**.

---

# 2. Goals

The project should demonstrate the ability to:

* prepare and validate ML data
* train a reproducible ML model
* perform hyperparameter optimization
* track experiments
* version models
* register models
* expose a prediction API
* containerize an ML application
* run the system in a Linux-like environment
* implement CI and automated testing
* deploy the service to cloud infrastructure
* monitor inference behavior
* detect data/model degradation
* retrain the model
* evaluate candidate models
* prevent an inferior model from automatically replacing production
* benchmark inference performance

---

# 3. Non-Goals

The first version must NOT attempt to build a large distributed platform.

Do not introduce the following unless a later phase explicitly requires them:

* Kubernetes
* Kafka
* Spark
* Ray
* Terraform
* Airflow
* service mesh
* distributed feature store
* large-scale data lake
* complex microservice architecture
* GPU cluster
* multi-region deployment

These technologies may be explored later, but they are not required for the core project.

The project should prioritize:

> **working system > architectural complexity**

---

# 4. Core ML Problem

## 4.1 Task

Predict whether a customer is likely to churn.

Input:

```text
Customer features
        ↓
ML model
        ↓
Probability of churn
```

Example:

```json
{
  "customer_id": "C12345",
  "tenure": 12,
  "monthly_charge": 89.5,
  "contract_type": "monthly",
  "payment_method": "credit_card",
  "support_tickets": 4,
  "usage_frequency": 0.42
}
```

Output:

```json
{
  "prediction": 1,
  "churn_probability": 0.87,
  "model_version": "churn-model-v3"
}
```

---

# 5. Dataset

## 5.1 Initial Dataset

Use a public customer churn dataset.

Preferred initial choice:

**Telco Customer Churn**

The dataset should contain:

* numerical features
* categorical features
* binary target
* realistic missing/invalid values
* enough observations to demonstrate a complete pipeline

Before implementation, verify:

* source
* license
* download method
* schema
* target definition
* class distribution

The exact dataset source must be documented in `README.md`.

---

# 6. ML Model

## 6.1 Primary Model

Use:

**XGBoost**

Reason:

* strong baseline for tabular data
* fast training
* easy deployment
* supports numerical features well
* supports hyperparameter tuning
* appropriate for a portfolio ML engineering project
* allows focus on ML infrastructure rather than expensive model training

---

# 7. ML Metrics

Accuracy must NOT be the only evaluation metric.

Primary metrics:

* PR-AUC
* Recall
* Precision
* F1
* ROC-AUC

Also report:

* confusion matrix
* class distribution
* threshold used for classification

## Primary Objective

The initial optimization objective should prioritize:

> **PR-AUC and recall**

because churn prediction can have class imbalance and missing a high-risk customer can be costly.

The exact production threshold must be configurable.

Do not hard-code the classification threshold inside the model implementation.

---

# 8. Data Pipeline

The data pipeline should contain the following stages:

```text
Raw Data
   ↓
Schema Validation
   ↓
Data Cleaning
   ↓
Train / Validation / Test Split
   ↓
Feature Engineering
   ↓
Model Training
```

The pipeline must prevent data leakage.

Important rules:

* preprocessing parameters must be learned only from training data
* validation/test data must not influence training
* train/validation/test split must be reproducible
* random seeds must be configurable
* transformations used during training must also be used during inference

---

# 9. Feature Engineering

Feature engineering should support:

* numerical features
* categorical features
* missing values
* categorical encoding
* numerical transformations where justified

The preprocessing pipeline must be serializable together with the model or otherwise versioned consistently.

Production inference must use exactly the same preprocessing logic as training.

Avoid manually duplicating preprocessing code between training and serving.

---

# 10. Hyperparameter Optimization

Use:

**Optuna**

The optimization process should search over a reasonable parameter space.

Potential parameters:

```text
max_depth
learning_rate
n_estimators
subsample
colsample_bytree
min_child_weight
```

The objective should be explicitly defined.

Example:

```text
maximize PR-AUC
```

or:

```text
maximize Recall
subject to Precision >= threshold
```

Do not perform arbitrary tuning simply to demonstrate Optuna.

Every experiment should have a measurable objective.

---

# 11. Experiment Tracking

Use:

**MLflow**

MLflow must track at minimum:

## Parameters

```text
model type
dataset version
random seed
hyperparameters
feature configuration
classification threshold
```

## Metrics

```text
PR-AUC
ROC-AUC
precision
recall
F1
```

## Artifacts

```text
trained model
evaluation report
confusion matrix
configuration
feature information
```

The goal is reproducibility.

A developer should be able to determine:

> Which configuration produced the production model?

---

# 12. Model Registry

Use MLflow Model Registry.

Model lifecycle:

```text
Candidate
    ↓
Evaluation
    ↓
Approved
    ↓
Production
    ↓
Archived
```

The system must associate a model with:

* model version
* training run
* parameters
* metrics
* dataset version
* creation timestamp

---

# 13. Model Quality Gate

A new model must NOT automatically replace the production model.

Example:

```text
Production Model
PR-AUC = 0.71

Candidate Model
PR-AUC = 0.75
```

→ Candidate may be promoted.

But:

```text
Production Model
PR-AUC = 0.71

Candidate Model
PR-AUC = 0.68
```

→ Candidate must be rejected.

The acceptance criteria must be explicit and configurable.

Example:

```text
candidate_pr_auc >= production_pr_auc
```

Additional constraints may be introduced later:

```text
candidate_recall >= production_recall
candidate_precision >= minimum_precision
```

---

# 14. Model Serving

Use:

**FastAPI**

The service must expose:

```text
GET  /health
GET  /model-info
POST /predict
```

## `/health`

Used to determine whether the service is alive and able to load a model.

Example:

```json
{
  "status": "healthy"
}
```

## `/model-info`

Returns:

```json
{
  "model_name": "churn-model",
  "model_version": "3"
}
```

## `/predict`

Input:

```json
{
  "features": {}
}
```

Output:

```json
{
  "prediction": 1,
  "churn_probability": 0.87,
  "model_version": "3"
}
```

Input validation must be implemented with Pydantic.

Invalid requests must return appropriate HTTP errors.

---

# 15. Model Loading

The model must NOT be loaded from disk for every request.

Preferred lifecycle:

```text
Application Startup
       ↓
Load Model
       ↓
Keep Model In Memory
       ↓
Handle Requests
```

The service should fail clearly if the configured production model cannot be loaded.

---

# 16. REST API Contract

The API contract must be documented.

At minimum:

* request schema
* response schema
* error responses
* model version behavior
* health behavior

FastAPI's generated OpenAPI documentation should be available.

---

# 17. Docker

The inference service must run inside Docker.

Docker requirements:

* use a lightweight Python base image where appropriate
* install pinned dependencies
* use environment variables for configuration
* do not hard-code secrets
* expose the application port
* provide a health check where appropriate
* run as a non-root user if practical

Expected flow:

```text
Source Code
    ↓
Dockerfile
    ↓
docker build
    ↓
Docker Image
    ↓
Container
    ↓
FastAPI
```

---

# 18. Docker Compose

Docker Compose may be used for local development.

Initial services should remain minimal.

Possible services:

```text
api
mlflow
```

A database should only be introduced if required by the selected MLflow deployment configuration.

Do not add Redis, Kafka, PostgreSQL, or other infrastructure without a concrete requirement.

---

# 19. Linux

The project must be Linux-compatible.

The development environment may be:

```text
Windows
   ↓
Docker Desktop
   ↓
Linux container
```

The production environment should assume:

```text
Linux-based cloud environment
```

The project documentation should include basic Linux commands for:

```bash
python
pip
curl
env
ps
grep
docker
```

The developer should understand:

* processes
* ports
* environment variables
* logs
* filesystem
* permissions
* container lifecycle

Linux is treated as the production runtime environment, not as a separate feature.

---

# 20. Configuration

Configuration must not be hard-coded.

Use environment variables and/or configuration files.

Examples:

```text
MODEL_NAME
MODEL_VERSION
MLFLOW_TRACKING_URI
API_HOST
API_PORT
LOG_LEVEL
```

Never commit:

* API keys
* cloud credentials
* passwords
* access tokens
* secrets

Provide `.env.example`.

---

# 21. Monitoring

Monitoring must cover both:

## 21.1 Application Metrics

Track:

```text
request count
error count
request latency
P50 latency
P95 latency
P99 latency
throughput
```

## 21.2 Model Metrics

Track:

```text
prediction distribution
predicted class distribution
probability/confidence distribution
model version
```

## 21.3 Data Quality

Monitor:

```text
missing values
invalid values
schema violations
feature ranges
unexpected categories
```

---

# 22. Data Drift

The system should detect changes between training/reference data and production data.

Potential methods:

* PSI
* KS test
* distribution comparison
* categorical frequency comparison

Example:

```text
Training:
monthly_charge
      ↓
Reference Distribution

Production:
monthly_charge
      ↓
Current Distribution

Compare
      ↓
Drift Score
```

The threshold must be configurable.

Do not claim that drift automatically means model performance has degraded.

Drift is a **signal**, not proof of model failure.

---

# 23. Retraining

Retraining should follow:

```text
New Data
   ↓
Validation
   ↓
Training
   ↓
Evaluation
   ↓
Compare With Production
   ↓
Quality Gate
   ↓
Promote / Reject
```

A retraining trigger may be:

* scheduled retraining
* data drift threshold
* performance degradation

The first implementation may use a manual or command-line trigger.

Automatic scheduling can be added later.

---

# 24. Retraining Safety

Retraining must never blindly replace the production model.

Example:

```text
Production
PR-AUC = 0.71

        ↓

Retrain

        ↓

Candidate
PR-AUC = 0.74

        ↓

Quality Gate

        ↓

PROMOTE
```

If:

```text
Candidate
PR-AUC = 0.65
```

then:

```text
REJECT
```

The production model remains unchanged.

---

# 25. CI

Use:

**GitHub Actions**

The initial CI pipeline should run:

```text
checkout
   ↓
install dependencies
   ↓
lint
   ↓
unit tests
```

Recommended tools:

* pytest
* ruff

CI must fail if tests fail.

Do not create a complex deployment pipeline until CI is stable.

---

# 26. Testing

Minimum tests:

## Data

* schema validation
* preprocessing
* missing-value handling

## Model

* model loads successfully
* prediction shape
* prediction range
* deterministic behavior where expected

## API

* `/health`
* `/model-info`
* `/predict`
* invalid input
* model unavailable behavior

## Quality Gate

Test:

```text
better candidate → accepted
worse candidate → rejected
```

---

# 27. Cloud Deployment

Target:

**AWS ECS Fargate**

Expected deployment architecture:

```text
Developer
    ↓
GitHub
    ↓
GitHub Actions
    ↓
Docker Build
    ↓
Amazon ECR
    ↓
Amazon ECS
    ↓
Fargate
    ↓
FastAPI Container
```

The deployment should expose the prediction API through an appropriate AWS networking configuration.

Cloud resources must be documented.

Avoid creating expensive resources unnecessarily.

---

# 28. CI/CD

Target lifecycle:

```text
Developer
    ↓
git push
    ↓
GitHub Actions
    ↓
Tests
    ↓
Docker Build
    ↓
Docker Image
    ↓
ECR
    ↓
Deployment
```

The first implementation may stop at:

```text
GitHub
 ↓
Tests
 ↓
Docker Build
```

Cloud deployment can be introduced after the local pipeline is stable.

---

# 29. Observability

Every prediction request should have enough metadata for debugging.

Recommended log fields:

```text
timestamp
request_id
model_version
latency_ms
prediction
```

Do not log unnecessary customer-sensitive information.

Logs should make it possible to answer:

> Which model produced this prediction?

and:

> How long did inference take?

---

# 30. Performance Benchmark

The project must include measurable performance experiments.

Measure:

```text
single request latency
batch latency
throughput
CPU usage
memory usage
```

Report:

```text
P50
P95
P99
requests/sec
```

Do not fabricate benchmark values.

Benchmark results must include:

* hardware/environment
* model version
* batch size
* number of requests
* warm-up behavior
* measurement method

---

# 31. Optional Deep Learning / GPU Extension

The core platform does NOT require GPU.

A later extension may introduce a PyTorch model to demonstrate GPU optimization.

Possible experiments:

```text
CPU vs GPU

Batch 1
Batch 16
Batch 32
Batch 64

FP32
vs
Mixed Precision
```

Measure:

```text
throughput
latency
GPU memory
training time
```

If no local NVIDIA GPU exists, use a suitable cloud environment.

Never fabricate GPU measurements.

---

# 32. Optional gRPC Extension

gRPC is NOT required for the first version.

If implemented, use it to demonstrate internal service-to-service communication.

Architecture:

```text
Client
   ↓ REST
FastAPI
   ↓ gRPC
Inference Service
   ↓
Model
```

The experiment should have a clear reason.

Possible benchmark:

```text
REST vs gRPC
```

Measure:

* latency
* throughput
* payload size

Do not add gRPC solely for technology keywords.

---

# 33. Microservice Strategy

The initial architecture should be a modular application rather than a collection of microservices.

Initial:

```text
ML Training
MLflow
FastAPI
```

Only split services when there is a concrete reason such as:

* independent scaling
* independent deployment
* resource isolation
* separate ownership
* different runtime requirements

Microservices are an architectural tool, not a project requirement.

---

# 34. Repository Structure

Recommended:

```text
ml-platform/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   ├── validation.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── engineering.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── tune.py
│   │   └── evaluate.py
│   │
│   ├── registry/
│   │   └── model_registry.py
│   │
│   ├── serving/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── model_loader.py
│   │
│   ├── monitoring/
│   │   ├── metrics.py
│   │   └── drift.py
│   │
│   └── retraining/
│       └── pipeline.py
│
├── tests/
│
├── configs/
│
├── scripts/
│
├── notebooks/
│
├── Dockerfile
├── compose.yaml
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── TECH.md
```

The exact structure may be adjusted if there is a simpler design.

Do not create empty modules solely to match this tree.

---

# 35. Development Phases

## Phase 1 — ML Baseline

Deliver:

* dataset
* preprocessing
* XGBoost model
* evaluation
* reproducible training

Success condition:

```text
Training can be executed from a clean environment.
```

---

## Phase 2 — Hyperparameter Optimization

Deliver:

* Optuna
* defined search space
* objective function
* best-model selection

Success condition:

```text
Optimization produces reproducible experiment results.
```

---

## Phase 3 — MLflow

Deliver:

* experiment tracking
* parameters
* metrics
* artifacts
* model registration

Success condition:

```text
Every training run can be inspected and reproduced.
```

---

## Phase 4 — Model Serving

Deliver:

* FastAPI
* `/health`
* `/model-info`
* `/predict`
* Pydantic validation
* model loaded at startup

Success condition:

```text
A trained model can serve predictions through HTTP.
```

---

## Phase 5 — Docker

Deliver:

* Dockerfile
* local container
* configuration through environment variables

Success condition:

```text
The API works inside a clean container.
```

---

## Phase 6 — Testing + CI

Deliver:

* pytest
* ruff
* GitHub Actions

Success condition:

```text
git push
   ↓
automated validation
```

---

## Phase 7 — Cloud Deployment

Deliver:

* ECR
* ECS
* Fargate
* running API

Success condition:

```text
External client
      ↓
Cloud API
      ↓
Production model
```

---

## Phase 8 — Monitoring

Deliver:

* latency
* request count
* error count
* model version
* prediction distribution

Success condition:

```text
Operator can determine whether the service is healthy.
```

---

## Phase 9 — Drift Detection

Deliver:

* reference dataset
* production data statistics
* drift calculation
* configurable threshold
* drift report

Success condition:

```text
The system can identify statistically significant distribution changes.
```

---

## Phase 10 — Retraining

Deliver:

* retraining command
* candidate model
* evaluation
* quality gate
* promotion/rejection

Success condition:

```text
A candidate model can safely replace production only when it satisfies defined criteria.
```

---

## Phase 11 — Performance Benchmark

Deliver:

* benchmark script
* latency measurements
* throughput measurements
* reproducible methodology

Success condition:

```text
Performance claims are backed by measured experiments.
```

---

## Phase 12 — Advanced Extensions

Optional:

* gRPC
* PyTorch
* GPU benchmark
* canary deployment
* automated scheduling
* advanced monitoring
* Kubernetes
* infrastructure as code

Only implement these if the core system is stable.

---

# 36. Definition of Done

The project is considered complete when:

### ML

* [ ] reproducible training
* [ ] preprocessing pipeline
* [ ] XGBoost model
* [ ] Optuna tuning
* [ ] meaningful evaluation metrics

### MLOps

* [ ] MLflow tracking
* [ ] model registry
* [ ] model versioning
* [ ] quality gate
* [ ] retraining workflow

### Serving

* [ ] FastAPI
* [ ] `/health`
* [ ] `/model-info`
* [ ] `/predict`
* [ ] input validation
* [ ] model loaded once

### Infrastructure

* [ ] Docker
* [ ] Docker Compose where useful
* [ ] Linux-compatible workflow
* [ ] environment-based configuration

### DevOps

* [ ] Git
* [ ] pytest
* [ ] ruff
* [ ] GitHub Actions

### Cloud

* [ ] ECR
* [ ] ECS Fargate
* [ ] deployed inference API

### Observability

* [ ] latency
* [ ] errors
* [ ] request count
* [ ] model version
* [ ] prediction distribution
* [ ] drift detection

### Performance

* [ ] latency benchmark
* [ ] throughput benchmark
* [ ] reproducible benchmark methodology

---

# 37. Engineering Principles

## Principle 1 — Solve the problem before adding technology

Never add a tool because it is popular.

Ask:

> What problem does this tool solve?

---

## Principle 2 — Prefer measurable behavior

Bad:

```text
"Added monitoring."
```

Good:

```text
"P95 inference latency is measured and exposed."
```

---

## Principle 3 — Reproducibility over convenience

Training should be reproducible through configuration and versioned dependencies.

---

## Principle 4 — Production safety

A new model must never replace production blindly.

---

## Principle 5 — Avoid technology collecting

The goal is not:

```text
XGBoost
MLflow
Docker
Kubernetes
Kafka
Spark
Ray
Terraform
Redis
gRPC
...
```

The goal is:

```text
Reliable ML lifecycle
```

---

## Principle 6 — Keep the architecture explainable

Every component must have a clear answer to:

> Why does this exist?

---

# 38. Agent Development Rules

Any coding agent working on this repository must follow these rules:

1. Read `TECH.md` before making architectural changes.
2. Do not introduce new infrastructure without justification.
3. Do not silently expand scope.
4. Do not fabricate benchmark results.
5. Do not fabricate model metrics.
6. Do not hard-code secrets.
7. Do not create unnecessary abstractions.
8. Prefer incremental implementation.
9. Run tests after meaningful changes.
10. Explain architectural changes.
11. Preserve reproducibility.
12. Update documentation when behavior changes.
13. Never replace the production model without passing the quality gate.
14. Keep training and serving preprocessing consistent.
15. Prefer a simple working implementation over a sophisticated incomplete one.

---

# 39. Final Architecture

The intended final architecture is:

```text
                         ┌─────────────────────┐
                         │     Dataset         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Validation     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Training + Optuna   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       MLflow        │
                         │ Tracking + Registry │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Quality Gate     │
                         └──────────┬──────────┘
                                    │
                           ┌────────┴────────┐
                           │                 │
                         Reject            Promote
                           │                 │
                           │                 ▼
                           │        ┌────────────────┐
                           │        │ Production     │
                           │        │ Model          │
                           │        └───────┬────────┘
                           │                │
                           │                ▼
                           │        ┌────────────────┐
                           │        │ FastAPI        │
                           │        │ Inference API  │
                           │        └───────┬────────┘
                           │                │
                           │                ▼
                           │        ┌────────────────┐
                           │        │ Docker / AWS   │
                           │        │ ECS Fargate    │
                           │        └───────┬────────┘
                           │                │
                           │                ▼
                           │        ┌────────────────┐
                           │        │ Monitoring     │
                           │        └───────┬────────┘
                           │                │
                           │        ┌───────┴────────┐
                           │        │                │
                           │       Healthy        Drift /
                           │                       Degradation
                           │                         │
                           │                         ▼
                           │                 ┌──────────────┐
                           │                 │ Retraining   │
                           │                 └──────┬───────┘
                           │                        │
                           │                        ▼
                           │                 Candidate Model
                           │                        │
                           │                        ▼
                           └──────────────── Quality Gate
```

---

# 40. Project Success Criterion

The project succeeds if an interviewer can ask:

> "What happens when your model is already deployed but the data changes?"

and the system has a concrete answer:

```text
Production traffic
      ↓
Monitoring
      ↓
Data / prediction drift detected
      ↓
Retraining
      ↓
Candidate model
      ↓
MLflow experiment
      ↓
Evaluation
      ↓
Quality Gate
      ↓
       ┌───────────────┐
       │               │
     Better          Worse
       │               │
       ▼               ▼
   Promote           Reject
       │
       ▼
Production
```

The project should demonstrate that the engineer understands not only:

> **"How do I train a model?"**

but also:

> **"How do I operate an ML model reliably after it has been trained?"**
