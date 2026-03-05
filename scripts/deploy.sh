#!/usr/bin/env bash
# ── First-time deployment script for Fraud Detection API ────
# Prerequisites: AWS CLI configured, Terraform installed, Docker running
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="fraud-detection"

echo "══════════════════════════════════════════════════════"
echo "  Fraud Detection — AWS ECS Deployment"
echo "══════════════════════════════════════════════════════"

# ── Step 1: Provision infrastructure ────────────────────────
echo ""
echo "▸ Step 1/4: Provisioning AWS infrastructure..."
cd infra
terraform init
terraform plan -out=tfplan
terraform apply tfplan

# Capture outputs
ECR_URL=$(terraform output -raw ecr_repository_url)
ALB_DNS=$(terraform output -raw alb_dns_name)
cd ..

# ── Step 2: Build & push Docker image ──────────────────────
echo ""
echo "▸ Step 2/4: Building and pushing Docker image..."
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$ECR_URL"

docker build --platform linux/amd64 -t "$APP_NAME" .
docker tag "$APP_NAME:latest" "$ECR_URL:latest"
docker push "$ECR_URL:latest"

# ── Step 3: Force new deployment ───────────────────────────
echo ""
echo "▸ Step 3/4: Deploying to ECS..."
aws ecs update-service \
  --cluster "${APP_NAME}-cluster" \
  --service "${APP_NAME}-service" \
  --force-new-deployment \
  --region "$AWS_REGION"

# ── Step 4: Wait for stability ─────────────────────────────
echo ""
echo "▸ Step 4/4: Waiting for service to stabilize..."
aws ecs wait services-stable \
  --cluster "${APP_NAME}-cluster" \
  --services "${APP_NAME}-service" \
  --region "$AWS_REGION"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ✓ Deployment complete!"
echo "  API endpoint: $ALB_DNS"
echo "  Health check: $ALB_DNS/health"
echo "  API docs:     $ALB_DNS/docs"
echo "══════════════════════════════════════════════════════"
