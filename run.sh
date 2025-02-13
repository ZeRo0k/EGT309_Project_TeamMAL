#!/bin/bash

# Define file paths for YAML files
PREPROCESSING_DEPLOYMENT="processing.yaml"
TRAINING_DEPLOYMENT="training.yaml"
OPTIMIZATION_DEPLOYMENT="optimization.yaml"

PREPROCESSING_SERVICE="preprocessing-service.yaml"
TRAINING_SERVICE="training-service.yaml"
OPTIMIZATION_SERVICE="optimization-service.yaml"

# Apply Kubernetes configurations
echo "Deploying Preprocessing Service..."
kubectl apply -f $PREPROCESSING_DEPLOYMENT
kubectl apply -f $PREPROCESSING_SERVICE
echo "Preprocessing Service Deployed!"

echo "Deploying Training Service..."
kubectl apply -f $TRAINING_DEPLOYMENT
kubectl apply -f $TRAINING_SERVICE
echo "Training Service Deployed!"

echo "Deploying Optimization Service..."
kubectl apply -f $OPTIMIZATION_DEPLOYMENT
kubectl apply -f $OPTIMIZATION_SERVICE
echo "Optimization Service Deployed!"

# Check the status of pods
echo "Checking Pod Status..."
kubectl get pods

# Verify the services
echo "Checking Service Status..."
kubectl get services

echo "All services and deployments are successfully applied!"
