.PHONY: help setup train test run simulate verify docker-build docker-run gcp-auth gke-credentials k8s-dry-run clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
UVICORN := .venv/bin/uvicorn

# GCP Configuration Defaults
GCP_PROJECT_ID ?= your-gcp-project-id
GKE_CLUSTER_NAME ?= mlops-gke-cluster
GKE_LOCATION ?= us-central1
GAR_REPOSITORY ?= mlops-repo
IMAGE_TAG ?= latest

help:
	@echo "======================================================================"
	@echo "🛡️  Real-Time Fraud Detection MLOps on Google Cloud (GKE + GAR)"
	@echo "======================================================================"
	@echo "make setup           - Install dependencies and create local environment"
	@echo "make train           - Execute ML model training & threshold calibration"
	@echo "make test            - Run pytest unit and integration test suite"
	@echo "make run             - Launch FastAPI server locally on http://localhost:8000"
	@echo "make verify          - Run end-to-end local validation checks"
	@echo "make simulate        - Stream synthetic live transaction traffic to API"
	@echo "make docker-build    - Build production Docker image locally"
	@echo "make docker-run      - Run Docker container on port 8000"
	@echo "make gcp-auth        - Configure local Docker with Google Artifact Registry"
	@echo "make gke-credentials - Fetch GKE cluster credentials for kubectl"
	@echo "make k8s-dry-run     - Validate all GKE Kubernetes manifests"
	@echo "make clean           - Remove caches and temporary files"
	@echo "======================================================================"

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt

train:
	$(PYTHON) pipeline/train.py

test:
	$(PYTEST) -v tests/

run:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

verify:
	$(PYTHON) scripts/verify_local.py

simulate:
	$(PYTHON) scripts/simulate_stream.py --count 30 --delay 0.05

docker-build:
	docker build -t $(GKE_LOCATION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GAR_REPOSITORY)/fraud-detector:$(IMAGE_TAG) .

docker-run:
	docker run --rm -p 8000:8000 --name fraud-api $(GKE_LOCATION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GAR_REPOSITORY)/fraud-detector:$(IMAGE_TAG)

gcp-auth:
	gcloud auth configure-docker $(GKE_LOCATION)-docker.pkg.dev --quiet

gke-credentials:
	gcloud container clusters get-credentials $(GKE_CLUSTER_NAME) --location $(GKE_LOCATION) --project $(GCP_PROJECT_ID)

k8s-dry-run:
	$(PYTHON) -c "import yaml, glob; [list(yaml.safe_load_all(open(p))) for p in glob.glob('k8s/*.yaml')]; print('All GKE manifests validated!')"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
