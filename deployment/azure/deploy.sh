#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Azure Deployment Script — AI Data Analyst Agent
# Run once to provision all Azure resources.
# Prerequisites: Azure CLI logged in, Docker installed.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration — edit these ───────────────────────────────────────────────
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-your-subscription-id}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-ai-data-analyst}"
LOCATION="${AZURE_LOCATION:-eastus}"
ACR_NAME="${AZURE_ACR_NAME:-acrdataanalyst}"
APP_SERVICE_PLAN="${AZURE_APP_SERVICE_PLAN:-asp-ai-data-analyst}"
WEBAPP_FRONTEND="${AZURE_WEBAPP_NAME:-ai-data-analyst}"
WEBAPP_BACKEND="${AZURE_WEBAPP_BACKEND:-ai-data-analyst-api}"
IMAGE_NAME="ai-data-analyst"
IMAGE_TAG="${1:-latest}"

echo "═══════════════════════════════════════════════════════════════"
echo " Azure Deployment: AI Data Analyst Agent"
echo " Resource Group : $RESOURCE_GROUP"
echo " Location       : $LOCATION"
echo " ACR             : $ACR_NAME"
echo " Image Tag       : $IMAGE_TAG"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. Set subscription ───────────────────────────────────────────────────────
az account set --subscription "$SUBSCRIPTION_ID"

# ── 2. Resource Group ─────────────────────────────────────────────────────────
echo "[1/7] Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# ── 3. Azure Container Registry ───────────────────────────────────────────────
echo "[2/7] Creating Azure Container Registry..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Standard \
  --admin-enabled true \
  --output none

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
echo "   ACR: $ACR_LOGIN_SERVER"

# ── 4. Build & push image ─────────────────────────────────────────────────────
echo "[3/7] Building and pushing Docker image..."
az acr login --name "$ACR_NAME"
docker build -t "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" .
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker tag "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"

# ── 5. App Service Plan ───────────────────────────────────────────────────────
echo "[4/7] Creating App Service Plan (B2)..."
az appservice plan create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_SERVICE_PLAN" \
  --is-linux \
  --sku B2 \
  --output none

# ── 6. Backend Web App ────────────────────────────────────────────────────────
echo "[5/7] Creating Backend Web App (FastAPI)..."
ACR_CREDS=$(az acr credential show --name "$ACR_NAME" -o json)
ACR_USER=$(echo "$ACR_CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['username'])")
ACR_PASS=$(echo "$ACR_CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['passwords'][0]['value'])")

az webapp create \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_SERVICE_PLAN" \
  --name "$WEBAPP_BACKEND" \
  --container-image-name "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
  --container-registry-url "https://$ACR_LOGIN_SERVER" \
  --container-registry-user "$ACR_USER" \
  --container-registry-password "$ACR_PASS" \
  --output none

az webapp config appsettings set \
  --resource-group "$RESOURCE_GROUP" \
  --name "$WEBAPP_BACKEND" \
  --settings \
    APP_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=4 \
    WEBSITES_PORT=8000 \
    GROQ_API_KEY="${GROQ_API_KEY:-}" \
  --output none

# ── 7. Frontend Web App ───────────────────────────────────────────────────────
echo "[6/7] Creating Frontend Web App (Streamlit)..."
BACKEND_URL="https://$WEBAPP_BACKEND.azurewebsites.net"

az webapp create \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_SERVICE_PLAN" \
  --name "$WEBAPP_FRONTEND" \
  --container-image-name "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
  --container-registry-url "https://$ACR_LOGIN_SERVER" \
  --container-registry-user "$ACR_USER" \
  --container-registry-password "$ACR_PASS" \
  --output none

az webapp config appsettings set \
  --resource-group "$RESOURCE_GROUP" \
  --name "$WEBAPP_FRONTEND" \
  --settings \
    APP_ENV=production \
    BACKEND_URL="$BACKEND_URL" \
    STREAMLIT_PORT=8501 \
    WEBSITES_PORT=8501 \
  --output none

echo "[7/7] Enabling managed identity & finalising..."
az webapp identity assign --name "$WEBAPP_BACKEND" --resource-group "$RESOURCE_GROUP" --output none
az webapp identity assign --name "$WEBAPP_FRONTEND" --resource-group "$RESOURCE_GROUP" --output none

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " ✅ Deployment complete!"
echo " Frontend : https://$WEBAPP_FRONTEND.azurewebsites.net"
echo " API      : https://$WEBAPP_BACKEND.azurewebsites.net"
echo " API Docs : https://$WEBAPP_BACKEND.azurewebsites.net/api/docs"
echo "═══════════════════════════════════════════════════════════════"
