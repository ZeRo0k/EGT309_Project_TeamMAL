#!/bin/bash

# Ensure the script has correct permissions
if [[ ! -x "$0" ]]; then
  echo "🔧 Setting executable permissions for run.sh..."
  chmod +x "$0"
fi

# ------------------- USER INPUT: DOCKER HUB USERNAME -------------------
read -p "Enter your Docker Hub username: " docker_username

# Validate Docker username input
if [[ -z "$docker_username" ]]; then
  echo "❌ Error: Docker username cannot be empty!"
  exit 1
fi

# ------------------- USER INPUT: PROJECT DIRECTORY PATH -------------------
read -p "Enter the full path where the project folder (EGT309_Project_TeamMAL-main) is located in your device: " project_path

# Validate project path input
if [[ ! -d "$project_path" ]]; then
  echo "❌ Error: The directory '$project_path' does not exist. Please check and try again."
  exit 1
fi

echo "✅ Using project directory: $project_path"

# ------------------- REPLACE PLACEHOLDERS IN YAML FILES -------------------
declare -a files=(
  "data_preprocessing/docker/job.yaml"
  "model_training/docker/job.yaml"
  "model_optimization/docker/job.yaml"
  "app/docker/deployment.yaml"
  "run.sh"
)

echo "🔄 Updating Docker Hub username in all configuration files..."
for file in "${files[@]}"; do
  sed -i "s|zerook2005|$docker_username|g" "$file"
done

echo "✅ Updated all files with Docker username: $docker_username"

# ------------------- START MINIKUBE -------------------
echo "🚀 Starting Minikube..."
minikube start --memory 4096 --cpus 2
eval $(minikube docker-env)

# ------------------- CREATE DIRECTORIES IN MINIKUBE -------------------
echo "📂 Creating directories in Minikube..."
minikube ssh -- "sudo mkdir -p /mnt/data/datasets/raw_datasets && sudo mkdir -p /mnt/data/datasets/cleaned_datasets && sudo mkdir -p /mnt/data/saved_model"

# ------------------- COPY DATASETS TO MINIKUBE -------------------
echo "📥 Copying datasets to Minikube..."
minikube cp "$project_path/datasets/raw_datasets/train.csv" /mnt/data/datasets/raw_datasets/train.csv

# ------------------- SET FILE PERMISSIONS -------------------
echo "🔧 Setting file permissions..."
minikube ssh -- "sudo chmod -R 777 /mnt/data/datasets/raw_datasets && sudo chmod -R 777 /mnt/data/datasets/cleaned_datasets && sudo chmod -R 777 /mnt/data/saved_model"

# ------------------- PROMPT: PULL IMAGES FROM DOCKER HUB -------------------
read -p "Do you want to pull existing images from Docker Hub? (y/n): " pull_choice
build_images=true  # Default to true, will be set to false if pulling succeeds

if [[ $pull_choice == "y" ]]; then
  echo "📥 Pulling Docker images from Docker Hub..."
  docker pull $docker_username/data-preprocessing:latest &&
  docker pull $docker_username/model-training:latest &&
  docker pull $docker_username/model-optimization:latest &&
  docker pull $docker_username/app:latest && build_images=false

  if [[ $build_images == false ]]; then
    echo "✅ Images pulled successfully. Skipping build process."
  else
    echo "⚠️ One or more images failed to pull. Proceeding with local build..."
  fi
else
  echo "⚠️ Skipping image pull. Proceeding with local builds..."
fi

# ------------------- BUILD DOCKER IMAGES IF NOT PULLED -------------------
if [[ $build_images == true ]]; then
  echo "🐳 Building Docker images..."
  docker build -t $docker_username/data-preprocessing:latest -f data_preprocessing/docker/Dockerfile .
  docker build -t $docker_username/model-training:latest -f model_training/docker/Dockerfile .
  docker build -t $docker_username/model-optimization:latest -f model_optimization/docker/Dockerfile .
  docker build -t $docker_username/app:latest -f app/docker/Dockerfile .
fi

# ------------------- PROMPT: PUSH IMAGES TO DOCKER HUB -------------------
read -p "Do you want to push the newly built images to Docker Hub? (y/n): " push_choice

if [[ $push_choice == "y" ]]; then
  echo "📤 Pushing Docker images to Docker Hub..."
  docker push $docker_username/data-preprocessing:latest
  docker push $docker_username/model-training:latest
  docker push $docker_username/model-optimization:latest
  docker push $docker_username/app:latest
else
  echo "⚠️ Skipping image push."
fi

# ------------------- APPLY KUBERNETES CONFIGURATIONS -------------------
echo "⚙️ Applying Kubernetes configurations..."
kubectl apply -f storage/pv_raw.yaml
kubectl apply -f storage/pvc_raw.yaml
kubectl apply -f storage/pv_cleaned.yaml
kubectl apply -f storage/pvc_cleaned.yaml
kubectl apply -f storage/pv_model.yaml
kubectl apply -f storage/pvc_model.yaml

# ------------------- DEPLOY CONTAINERS IN KUBERNETES -------------------
kubectl apply -f data_preprocessing/docker/job.yaml
kubectl apply -f model_training/docker/job.yaml
kubectl apply -f model_optimization/docker/job.yaml
kubectl apply -f app/docker/deployment.yaml
kubectl apply -f app/docker/service.yaml

# ------------------- STREAM JOB OUTPUTS (LOGS) IN REAL-TIME -------------------
echo "📜 Streaming logs for data preprocessing..."
kubectl wait --for=condition=complete job/data-preprocessing-job --timeout=300s && \
kubectl logs -f job/data-preprocessing-job

echo "📜 Streaming logs for model training..."
kubectl wait --for=condition=complete job/model-training-job --timeout=300s && \
kubectl logs -f job/model-training-job

echo "📜 Streaming logs for model optimization..."
kubectl wait --for=condition=complete job/model-optimization-job --timeout=300s && \
kubectl logs -f job/model-optimization-job

# ------------------- OPTIONAL: SCALING, SELF-HEALING & ROLLBACK CHECK -------------------
read -p "Do you want to check scaling, self-healing, and perform rollout/rollback operations? (y/n): " scale_choice
if [[ $scale_choice == "y" ]]; then
  echo "📊 Checking Scaling, Self-Healing, Rollout, and Rollback capabilities..."

  # 1️⃣ Check Scaling
  echo "🔍 Checking running pods for Streamlit deployment..."
  kubectl get pods

  # 2️⃣ Check Self-Healing by Deleting a Pod
  echo "🛠 Testing Self-Healing: Deleting one Streamlit pod..."
  POD_NAME=$(kubectl get pods -l app=streamlit-app -o jsonpath="{.items[0].metadata.name}")
  kubectl delete pod $POD_NAME
  echo "⏳ Waiting for Kubernetes to self-heal..."
  sleep 10
  echo "✅ Self-healing test completed. Checking pod status..."
  kubectl get pods

  # 3️⃣ Perform Rollout and Rollback Testing
  echo "🚀 Checking rollout update..."
  kubectl rollout status deployment/streamlit-app
  kubectl rollout history deployment/streamlit-app

  REVISION_COUNT=$(kubectl rollout history deployment/streamlit-app | wc -l)

  if [[ $REVISION_COUNT -le 2 ]]; then
    echo "⚠️ Only one deployment revision exists. Creating a new revision..."
    
    # Update deployment to create a new revision
    NEW_IMAGE="zerook2005/app:v2"
    echo "🔄 Updating deployment image to: $NEW_IMAGE"
    kubectl set image deployment/streamlit-app streamlit-app=$NEW_IMAGE
    
    # Annotate for tracking rollback history
    kubectl annotate deployment streamlit-app kubernetes.io/change-cause="Updated to v2"

    # Wait for the new rollout to complete
    kubectl rollout status deployment/streamlit-app
  fi

  # Now we have two revisions, proceed with rollback
  echo "⏪ Rolling back to previous version..."
  kubectl rollout undo deployment/streamlit-app --to-revision=1
  kubectl rollout status deployment/streamlit-app
  kubectl rollout history deployment/streamlit-app
  kubectl get pods

else
  echo "⏩ Skipping scaling, self-healing, and rollout operations."
fi

# ------------------- CHECK POD & SERVICE STATUS -------------------
echo "📡 Checking Kubernetes resources..."
kubectl get pods
kubectl get services

# ------------------- WAIT FOR STREAMLIT APP TO BE READY -------------------
echo "⏳ Waiting for Streamlit app to be ready..."
kubectl wait --for=condition=ready pod --selector=app=streamlit-app --timeout=120s

# ------------------- PORT FORWARDING TO ACCESS STREAMLIT APP -------------------
echo "🔗 Port forwarding Streamlit app to localhost:8501..."
kubectl port-forward service/streamlit-service 8501:8501 &

echo "✅ Deployment completed! Open the app at: http://127.0.0.1:8501"
