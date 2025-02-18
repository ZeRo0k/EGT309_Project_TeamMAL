#!/bin/bash

# Start Minikube and configure Docker
echo "Starting Minikube..."
minikube start --memory 4096 --cpus 2
eval $(minikube docker-env)

# Create necessary directories in Minikube
echo "Creating directories in Minikube..."
minikube ssh -- "sudo mkdir -p /mnt/data/datasets/raw_datasets && sudo mkdir -p /mnt/data/datasets/cleaned_datasets && sudo mkdir -p /mnt/data/saved_model"

# Copy datasets into Minikube
echo "Copying datasets to Minikube..."
minikube cp /mnt/c/Users/User/OneDrive/Documents/NYP_Year_3/EGT309/MAL_Project/MAL_Project/datasets/raw_datasets/train.csv /mnt/data/datasets/raw_datasets/train.csv

# Give full permissions for files
minikube ssh -- "sudo chmod -R 777 /mnt/data/datasets/raw_datasets && sudo chmod -R 777 /mnt/data/datasets/cleaned_datasets && sudo chmod -R 777 /mnt/data/saved_model"

# Build Docker images
echo "Building Docker images..."
docker build -t zerook2005/data-preprocessing:latest -f data_preprocessing/docker/Dockerfile .
docker build -t zerook2005/model-training:latest -f model_training/docker/Dockerfile .
docker build -t zerook2005/model-optimization:latest -f model_optimization/docker/Dockerfile .
docker build -t zerook2005/app:latest -f app/docker/Dockerfile .

echo "Pushing Docker images to Docker Hub..."
docker push zerook2005/data-preprocessing:latest
docker push zerook2005/model-training:latest
docker push zerook2005/model-optimization:latest
docker push zerook2005/app:latest

# Apply Kubernetes PVs/PVCs
echo "Applying Kubernetes configurations..."
kubectl apply -f storage/pv_raw.yaml
kubectl apply -f storage/pvc_raw.yaml
kubectl apply -f storage/pv_cleaned.yaml
kubectl apply -f storage/pvc_cleaned.yaml
kubectl apply -f storage/pv_model.yaml
kubectl apply -f storage/pvc_model.yaml

# Apply Kubernetes Jobs/Deployments
kubectl apply -f data_preprocessing/docker/job.yaml
kubectl apply -f model_training/docker/job.yaml
kubectl apply -f model_optimization/docker/job.yaml
kubectl apply -f app/docker/deployment.yaml
kubectl apply -f app/docker/service.yaml

# Optional Kubernetes Scaling, Rollouts, and Rollbacks (for presentation)
read -p "Do you want to perform scaling or rollout operations now? (y/n): " answer
if [[ $answer == "y" ]]; then
  echo "Adding scaling, rollout, and rollback capabilities..."
  kubectl scale deployment streamlit-app --replicas=3
  kubectl rollout status deployment streamlit-app
  kubectl rollout history deployment streamlit-app
  kubectl rollout undo deployment streamlit-app
else
  echo "Skipping scaling and rollout operations."
fi

# Check the status of pods and services
echo "Checking status of Kubernetes resources..."
kubectl get pods
kubectl get services

# Wait for Streamlit pods to be ready
echo "Waiting for Streamlit pods to become ready..."
kubectl wait --for=condition=ready pod --selector=app=streamlit-app --timeout=120s

# Port forward the Streamlit app
echo "Port forwarding Streamlit app to localhost:8501..."
kubectl port-forward service/streamlit-service 8501:8501 &

echo "Minikube setup, Docker image build, and application deployment completed!"
echo "Open URL below in your browser"
echo "http://127.0.0.1:8501"


