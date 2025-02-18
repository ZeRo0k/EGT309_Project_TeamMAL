# # # # # # #!/bin/bash

# # # # # # # Start Minikube and configure Docker
# # # # # # echo "Starting Minikube..."
# # # # # # minikube start --memory 4096 --cpus 2
# # # # # # eval $(minikube docker-env)

# # # # # # # Create necessary directories in Minikube
# # # # # # echo "Creating directories in Minikube..."
# # # # # # minikube ssh -- "sudo mkdir -p /mnt/data/datasets/raw_datasets && sudo mkdir -p /mnt/data/datasets/cleaned_datasets && sudo mkdir -p /mnt/data/saved_model"

# # # # # # # Copy datasets into Minikube
# # # # # # echo "Copying datasets to Minikube..."
# # # # # # minikube cp /mnt/c/Users/User/OneDrive/Documents/NYP_Year_3/EGT309/MAL_Project/MAL_Project/datasets/raw_datasets/train.csv /mnt/data/datasets/raw_datasets/train.csv

# # # # # # # Give full permissions for files
# # # # # # minikube ssh -- "sudo chmod -R 777 /mnt/data/datasets/raw_datasets && sudo chmod -R 777 /mnt/data/datasets/cleaned_datasets && sudo chmod -R 777 /mnt/data/saved_model"

# # # # # # # Build Docker images
# # # # # # echo "Building Docker images..."
# # # # # # docker build -t zerook2005/data-preprocessing:latest -f data_preprocessing/docker/Dockerfile .
# # # # # # docker build -t zerook2005/model-training:latest -f model_training/docker/Dockerfile .
# # # # # # docker build -t zerook2005/model-optimization:latest -f model_optimization/docker/Dockerfile .
# # # # # # docker build -t zerook2005/app:latest -f app/docker/Dockerfile .

# # # # # # echo "Pushing Docker images to Docker Hub..."
# # # # # # docker push zerook2005/data-preprocessing:latest
# # # # # # docker push zerook2005/model-training:latest
# # # # # # docker push zerook2005/model-optimization:latest
# # # # # # docker push zerook2005/app:latest

# # # # # # # Apply Kubernetes PVs/PVCs
# # # # # # echo "Applying Kubernetes configurations..."
# # # # # # kubectl apply -f storage/pv_raw.yaml
# # # # # # kubectl apply -f storage/pvc_raw.yaml
# # # # # # kubectl apply -f storage/pv_cleaned.yaml
# # # # # # kubectl apply -f storage/pvc_cleaned.yaml
# # # # # # kubectl apply -f storage/pv_model.yaml
# # # # # # kubectl apply -f storage/pvc_model.yaml

# # # # # # # Apply Kubernetes Jobs/Deployments
# # # # # # kubectl apply -f data_preprocessing/docker/job.yaml
# # # # # # kubectl apply -f model_training/docker/job.yaml
# # # # # # kubectl apply -f model_optimization/docker/job.yaml
# # # # # # kubectl apply -f app/docker/deployment.yaml
# # # # # # kubectl apply -f app/docker/service.yaml

# # # # # # # Optional Kubernetes Scaling, Rollouts, and Rollbacks (for presentation)
# # # # # # read -p "Do you want to perform scaling or rollout operations now? (y/n): " answer
# # # # # # if [[ $answer == "y" ]]; then
# # # # # #   echo "Adding scaling, rollout, and rollback capabilities..."
# # # # # #   kubectl scale deployment streamlit-app --replicas=3
# # # # # #   kubectl rollout status deployment streamlit-app
# # # # # #   kubectl rollout history deployment streamlit-app
# # # # # #   kubectl rollout undo deployment streamlit-app
# # # # # # else
# # # # # #   echo "Skipping scaling and rollout operations."
# # # # # # fi

# # # # # # # Check the status of pods and services
# # # # # # echo "Checking status of Kubernetes resources..."
# # # # # # kubectl get pods
# # # # # # kubectl get services

# # # # # # # Wait for Streamlit pods to be ready
# # # # # # echo "Waiting for Streamlit pods to become ready..."
# # # # # # kubectl wait --for=condition=ready pod --selector=app=streamlit-app --timeout=120s

# # # # # # # Port forward the Streamlit app
# # # # # # echo "Port forwarding Streamlit app to localhost:8501..."
# # # # # # kubectl port-forward service/streamlit-service 8501:8501 &

# # # # # # echo "Minikube setup, Docker image build, and application deployment completed!"
# # # # # # echo "Open URL below in your browser"
# # # # # # echo "http://127.0.0.1:8501"






# # # # # change to prompt user to insert docker username
# # # # # #!/bin/bash

# # # # # # Ask user for Docker Hub username
# # # # # read -p "Enter your Docker Hub username: " docker_username

# # # # # # Validate input
# # # # # if [[ -z "$docker_username" ]]; then
# # # # #   echo "❌ Error: Docker username cannot be empty!"
# # # # #   exit 1
# # # # # fi

# # # # # # Define file paths for replacement
# # # # # declare -a files=(
# # # # #   "data_preprocessing/docker/job.yaml"
# # # # #   "model_training/docker/job.yaml"
# # # # #   "model_optimization/docker/job.yaml"
# # # # #   "app/docker/deployment.yaml"
# # # # #   "run.sh"
# # # # # )

# # # # # # Replace username placeholder `thisisangelz` in all files
# # # # # echo "Updating Docker Hub username in all configuration files..."
# # # # # for file in "${files[@]}"; do
# # # # #   sed -i "s|\thisisangelz|$docker_username|g" "$file"
# # # # # done

# # # # # echo "✅ Updated all files with Docker username: $docker_username"

# # # # # # Continue with Docker build and Kubernetes deployment...

# # # # # # Start Minikube and configure Docker
# # # # # echo "Starting Minikube..."
# # # # # minikube start --memory 4096 --cpus 2
# # # # # eval $(minikube docker-env)

# # # # # # Create necessary directories in Minikube
# # # # # echo "Creating directories in Minikube..."
# # # # # minikube ssh -- "sudo mkdir -p /mnt/data/datasets/raw_datasets && sudo mkdir -p /mnt/data/datasets/cleaned_datasets && sudo mkdir -p /mnt/data/saved_model"

# # # # # # Copy datasets into Minikube
# # # # # echo "Copying datasets to Minikube..."
# # # # # minikube cp /mnt/c/Users/User/OneDrive/Documents/NYP_Year_3/EGT309/MAL_Project/MAL_Project/datasets/raw_datasets/train.csv /mnt/data/datasets/raw_datasets/train.csv

# # # # # # Give full permissions for files
# # # # # minikube ssh -- "sudo chmod -R 777 /mnt/data/datasets/raw_datasets && sudo chmod -R 777 /mnt/data/datasets/cleaned_datasets && sudo chmod -R 777 /mnt/data/saved_model"

# # # # # # Build Docker images
# # # # # echo "Building Docker images..."
# # # # # docker build -t $docker_username/data-preprocessing:latest -f data_preprocessing/docker/Dockerfile .
# # # # # docker build -t $docker_username/model-training:latest -f model_training/docker/Dockerfile .
# # # # # docker build -t $docker_username/model-optimization:latest -f model_optimization/docker/Dockerfile .
# # # # # docker build -t $docker_username/app:latest -f app/docker/Dockerfile .

# # # # # echo "Pushing Docker images to Docker Hub..."
# # # # # docker push $docker_username/data-preprocessing:latest
# # # # # docker push $docker_username/model-training:latest
# # # # # docker push $docker_username/model-optimization:latest
# # # # # docker push $docker_username/app:latest

# # # # # # Apply Kubernetes PVs/PVCs
# # # # # echo "Applying Kubernetes configurations..."
# # # # # kubectl apply -f storage/pv_raw.yaml
# # # # # kubectl apply -f storage/pvc_raw.yaml
# # # # # kubectl apply -f storage/pv_cleaned.yaml
# # # # # kubectl apply -f storage/pvc_cleaned.yaml
# # # # # kubectl apply -f storage/pv_model.yaml
# # # # # kubectl apply -f storage/pvc_model.yaml

# # # # # # Apply Kubernetes Jobs/Deployments
# # # # # kubectl apply -f data_preprocessing/docker/job.yaml
# # # # # kubectl apply -f model_training/docker/job.yaml
# # # # # kubectl apply -f model_optimization/docker/job.yaml
# # # # # kubectl apply -f app/docker/deployment.yaml
# # # # # kubectl apply -f app/docker/service.yaml

# # # # # # Optional Kubernetes Scaling, Rollouts, and Rollbacks (for presentation)
# # # # # read -p "Do you want to perform scaling or rollout operations now? (y/n): " answer
# # # # # if [[ $answer == "y" ]]; then
# # # # #   echo "Adding scaling, rollout, and rollback capabilities..."
# # # # #   kubectl scale deployment streamlit-app --replicas=3
# # # # #   kubectl rollout status deployment streamlit-app
# # # # #   kubectl rollout history deployment streamlit-app
# # # # #   kubectl rollout undo deployment streamlit-app
# # # # # else
# # # # #   echo "Skipping scaling and rollout operations."
# # # # # fi

# # # # # # Check the status of pods and services
# # # # # echo "Checking status of Kubernetes resources..."
# # # # # kubectl get pods
# # # # # kubectl get services

# # # # # # Wait for Streamlit pods to be ready
# # # # # echo "Waiting for Streamlit pods to become ready..."
# # # # # kubectl wait --for=condition=ready pod --selector=app=streamlit-app --timeout=120s

# # # # # # Port forward the Streamlit app
# # # # # echo "Port forwarding Streamlit app to localhost:8501..."
# # # # # kubectl port-forward service/streamlit-service 8501:8501 &

# # # # # echo "✅ Minikube setup, Docker image build, and application deployment completed!"
# # # # # echo "🔗 Open URL below in your browser"
# # # # # echo "http://127.0.0.1:8501"






# # # # # # prompt the user to insert project folder path too
# # # # # #!/bin/bash

# # # # # # ------------------- USER INPUT: DOCKER HUB USERNAME -------------------
# # # # # read -p "Enter your Docker Hub username: " docker_username

# # # # # # Validate Docker username input
# # # # # if [[ -z "$docker_username" ]]; then
# # # # #   echo "❌ Error: Docker username cannot be empty!"
# # # # #   exit 1
# # # # # fi

# # # # # # ------------------- USER INPUT: PROJECT DIRECTORY PATH -------------------
# # # # # read -p "Enter the full path where the project folder (EGT309_Project_TeamMAL-main) is located: " project_path

# # # # # # Validate project path input
# # # # # if [[ ! -d "$project_path" ]]; then
# # # # #   echo "❌ Error: The directory '$project_path' does not exist. Please check and try again."
# # # # #   exit 1
# # # # # fi

# # # # # echo "✅ Using project directory: $project_path"

# # # # # # ------------------- REPLACE PLACEHOLDERS IN YAML FILES -------------------
# # # # # declare -a files=(
# # # # #   "data_preprocessing/docker/job.yaml"
# # # # #   "model_training/docker/job.yaml"
# # # # #   "model_optimization/docker/job.yaml"
# # # # #   "app/docker/deployment.yaml"
# # # # #   "run.sh"
# # # # # )

# # # # # echo "🔄 Updating Docker Hub username in all configuration files..."
# # # # # for file in "${files[@]}"; do
# # # # #   sed -i "s|thisisangelz|$docker_username|g" "$file"
# # # # # done

# # # # # echo "✅ Updated all files with Docker username: $docker_username"

# # # # # # ------------------- START MINIKUBE -------------------
# # # # # echo "🚀 Starting Minikube..."
# # # # # minikube start --memory 4096 --cpus 2
# # # # # eval $(minikube docker-env)

# # # # # # ------------------- CREATE DIRECTORIES IN MINIKUBE -------------------
# # # # # echo "📂 Creating directories in Minikube..."
# # # # # minikube ssh -- "sudo mkdir -p /mnt/data/datasets/raw_datasets && sudo mkdir -p /mnt/data/datasets/cleaned_datasets && sudo mkdir -p /mnt/data/saved_model"

# # # # # # ------------------- COPY DATASETS TO MINIKUBE -------------------
# # # # # echo "📥 Copying datasets to Minikube..."
# # # # # minikube cp "$project_path/datasets/raw_datasets/train.csv" /mnt/data/datasets/raw_datasets/train.csv

# # # # # # ------------------- SET FILE PERMISSIONS -------------------
# # # # # echo "🔧 Setting file permissions..."
# # # # # minikube ssh -- "sudo chmod -R 777 /mnt/data/datasets/raw_datasets && sudo chmod -R 777 /mnt/data/datasets/cleaned_datasets && sudo chmod -R 777 /mnt/data/saved_model"

# # # # # # ------------------- BUILD & PUSH DOCKER IMAGES -------------------
# # # # # echo "🐳 Building Docker images..."
# # # # # docker build -t $docker_username/data-preprocessing:latest -f data_preprocessing/docker/Dockerfile .
# # # # # docker build -t $docker_username/model-training:latest -f model_training/docker/Dockerfile .
# # # # # docker build -t $docker_username/model-optimization:latest -f model_optimization/docker/Dockerfile .
# # # # # docker build -t $docker_username/app:latest -f app/docker/Dockerfile .

# # # # # echo "📤 Pushing Docker images to Docker Hub..."
# # # # # docker push $docker_username/data-preprocessing:latest
# # # # # docker push $docker_username/model-training:latest
# # # # # docker push $docker_username/model-optimization:latest
# # # # # docker push $docker_username/app:latest

# # # # # # ------------------- APPLY KUBERNETES CONFIGURATIONS -------------------
# # # # # echo "⚙️ Applying Kubernetes configurations..."
# # # # # kubectl apply -f storage/pv_raw.yaml
# # # # # kubectl apply -f storage/pvc_raw.yaml
# # # # # kubectl apply -f storage/pv_cleaned.yaml
# # # # # kubectl apply -f storage/pvc_cleaned.yaml
# # # # # kubectl apply -f storage/pv_model.yaml
# # # # # kubectl apply -f storage/pvc_model.yaml

# # # # # # ------------------- DEPLOY CONTAINERS IN KUBERNETES -------------------
# # # # # kubectl apply -f data_preprocessing/docker/job.yaml
# # # # # kubectl apply -f model_training/docker/job.yaml
# # # # # kubectl apply -f model_optimization/docker/job.yaml
# # # # # kubectl apply -f app/docker/deployment.yaml
# # # # # kubectl apply -f app/docker/service.yaml

# # # # # # ------------------- OPTIONAL: SCALING & ROLLBACK -------------------
# # # # # read -p "Do you want to perform scaling or rollout operations now? (y/n): " answer
# # # # # if [[ $answer == "y" ]]; then
# # # # #   echo "📊 Adding scaling, rollout, and rollback capabilities..."
# # # # #   kubectl scale deployment streamlit-app --replicas=3
# # # # #   kubectl rollout status deployment streamlit-app
# # # # #   kubectl rollout history deployment streamlit-app
# # # # #   kubectl rollout undo deployment streamlit-app
# # # # # else
# # # # #   echo "⏩ Skipping scaling and rollout operations."
# # # # # fi

# # # # # # ------------------- CHECK POD & SERVICE STATUS -------------------
# # # # # echo "📡 Checking Kubernetes resources..."
# # # # # kubectl get pods
# # # # # kubectl get services

# # # # # # ------------------- WAIT FOR STREAMLIT APP TO BE READY -------------------
# # # # # echo "⏳ Waiting for Streamlit app to be ready..."
# # # # # kubectl wait --for=condition=ready pod --selector=app=streamlit-app --timeout=120s

# # # # # # ------------------- PORT FORWARDING TO ACCESS STREAMLIT APP -------------------
# # # # # echo "🔗 Port forwarding Streamlit app to localhost:8501..."
# # # # # kubectl port-forward service/streamlit-service 8501:8501 &

# # # # # echo "✅ Deployment completed! Open the app at: http://127.0.0.1:8501"








# # # # # added push/pull prompt
# # # # #!/bin/bash

# # # # # ------------------- USER INPUT: DOCKER HUB USERNAME -------------------
# # # # read -p "Enter your Docker Hub username: " docker_username

# # # # # Validate Docker username input
# # # # if [[ -z "$docker_username" ]]; then
# # # #   echo "❌ Error: Docker username cannot be empty!"
# # # #   exit 1
# # # # fi

# # # # # ------------------- USER INPUT: PROJECT DIRECTORY PATH -------------------
# # # # read -p "Enter the full path where the project folder (EGT309_Project_TeamMAL-main) is located: " project_path

# # # # # Validate project path input
# # # # if [[ ! -d "$project_path" ]]; then
# # # #   echo "❌ Error: The directory '$project_path' does not exist. Please check and try again."
# # # #   exit 1
# # # # fi

# # # # echo "✅ Using project directory: $project_path"

# # # # # ------------------- REPLACE PLACEHOLDERS IN YAML FILES -------------------
# # # # declare -a files=(
# # # #   "data_preprocessing/docker/job.yaml"
# # # #   "model_training/docker/job.yaml"
# # # #   "model_optimization/docker/job.yaml"
# # # #   "app/docker/deployment.yaml"
# # # #   "run.sh"
# # # # )

# # # # echo "🔄 Updating Docker Hub username in all configuration files..."
# # # # for file in "${files[@]}"; do
# # # #   sed -i "s|thisisangelz|$docker_username|g" "$file"
# # # # done

# # # # echo "✅ Updated all files with Docker username: $docker_username"

# # # # # ------------------- START MINIKUBE -------------------
# # # # echo "🚀 Starting Minikube..."
# # # # minikube start --memory 4096 --cpus 2
# # # # eval $(minikube docker-env)

# # # # # ------------------- CREATE DIRECTORIES IN MINIKUBE -------------------
# # # # echo "📂 Creating directories in Minikube..."
# # # # minikube ssh -- "sudo mkdir -p /mnt/data/datasets/raw_datasets && sudo mkdir -p /mnt/data/datasets/cleaned_datasets && sudo mkdir -p /mnt/data/saved_model"

# # # # # ------------------- COPY DATASETS TO MINIKUBE -------------------
# # # # echo "📥 Copying datasets to Minikube..."
# # # # minikube cp "$project_path/datasets/raw_datasets/train.csv" /mnt/data/datasets/raw_datasets/train.csv

# # # # # ------------------- SET FILE PERMISSIONS -------------------
# # # # echo "🔧 Setting file permissions..."
# # # # minikube ssh -- "sudo chmod -R 777 /mnt/data/datasets/raw_datasets && sudo chmod -R 777 /mnt/data/datasets/cleaned_datasets && sudo chmod -R 777 /mnt/data/saved_model"

# # # # # ------------------- PROMPT: PULL IMAGES FROM DOCKER HUB -------------------
# # # # read -p "Do you want to pull existing images from Docker Hub? (y/n): " pull_choice

# # # # if [[ $pull_choice == "y" ]]; then
# # # #   echo "📥 Pulling Docker images from Docker Hub..."
# # # #   docker pull $docker_username/data-preprocessing:latest
# # # #   docker pull $docker_username/model-training:latest
# # # #   docker pull $docker_username/model-optimization:latest
# # # #   docker pull $docker_username/app:latest
# # # # else
# # # #   echo "⚠️ Skipping image pull. Proceeding with local builds..."
# # # # fi

# # # # # ------------------- BUILD DOCKER IMAGES -------------------
# # # # echo "🐳 Building Docker images..."
# # # # docker build -t $docker_username/data-preprocessing:latest -f data_preprocessing/docker/Dockerfile .
# # # # docker build -t $docker_username/model-training:latest -f model_training/docker/Dockerfile .
# # # # docker build -t $docker_username/model-optimization:latest -f model_optimization/docker/Dockerfile .
# # # # docker build -t $docker_username/app:latest -f app/docker/Dockerfile .

# # # # # ------------------- PROMPT: PUSH IMAGES TO DOCKER HUB -------------------
# # # # read -p "Do you want to push the newly built images to Docker Hub? (y/n): " push_choice

# # # # if [[ $push_choice == "y" ]]; then
# # # #   echo "📤 Pushing Docker images to Docker Hub..."
# # # #   docker push $docker_username/data-preprocessing:latest
# # # #   docker push $docker_username/model-training:latest
# # # #   docker push $docker_username/model-optimization:latest
# # # #   docker push $docker_username/app:latest
# # # # else
# # # #   echo "⚠️ Skipping image push."
# # # # fi

# # # # # ------------------- APPLY KUBERNETES CONFIGURATIONS -------------------
# # # # echo "⚙️ Applying Kubernetes configurations..."
# # # # kubectl apply -f storage/pv_raw.yaml
# # # # kubectl apply -f storage/pvc_raw.yaml
# # # # kubectl apply -f storage/pv_cleaned.yaml
# # # # kubectl apply -f storage/pvc_cleaned.yaml
# # # # kubectl apply -f storage/pv_model.yaml
# # # # kubectl apply -f storage/pvc_model.yaml

# # # # # ------------------- DEPLOY CONTAINERS IN KUBERNETES -------------------
# # # # kubectl apply -f data_preprocessing/docker/job.yaml
# # # # kubectl apply -f model_training/docker/job.yaml
# # # # kubectl apply -f model_optimization/docker/job.yaml
# # # # kubectl apply -f app/docker/deployment.yaml
# # # # kubectl apply -f app/docker/service.yaml

# # # # # ------------------- OPTIONAL: SCALING & ROLLBACK -------------------
# # # # read -p "Do you want to perform scaling or rollout operations now? (y/n): " scale_choice
# # # # if [[ $scale_choice == "y" ]]; then
# # # #   echo "📊 Adding scaling, rollout, and rollback capabilities..."
# # # #   kubectl scale deployment streamlit-app --replicas=3
# # # #   kubectl rollout status deployment streamlit-app
# # # #   kubectl rollout history deployment streamlit-app
# # # #   kubectl rollout undo deployment streamlit-app
# # # # else
# # # #   echo "⏩ Skipping scaling and rollout operations."
# # # # fi

# # # # # ------------------- CHECK POD & SERVICE STATUS -------------------
# # # # echo "📡 Checking Kubernetes resources..."
# # # # kubectl get pods
# # # # kubectl get services

# # # # # ------------------- WAIT FOR STREAMLIT APP TO BE READY -------------------
# # # # echo "⏳ Waiting for Streamlit app to be ready..."
# # # # kubectl wait --for=condition=ready pod --selector=app=streamlit-app --timeout=120s

# # # # # ------------------- PORT FORWARDING TO ACCESS STREAMLIT APP -------------------
# # # # echo "🔗 Port forwarding Streamlit app to localhost:8501..."
# # # # kubectl port-forward service/streamlit-service 8501:8501 &

# # # # echo "✅ Deployment completed! Open the app at: http://127.0.0.1:8501"












# # # # added display logs (output)
# # # #!/bin/bash

# # # # ------------------- USER INPUT: DOCKER HUB USERNAME -------------------
# # # read -p "Enter your Docker Hub username: " docker_username

# # # # Validate Docker username input
# # # if [[ -z "$docker_username" ]]; then
# # #   echo "❌ Error: Docker username cannot be empty!"
# # #   exit 1
# # # fi

# # # # ------------------- USER INPUT: PROJECT DIRECTORY PATH -------------------
# # # read -p "Enter the full path where the project folder (EGT309_Project_TeamMAL-main) is located: " project_path

# # # # Validate project path input
# # # if [[ ! -d "$project_path" ]]; then
# # #   echo "❌ Error: The directory '$project_path' does not exist. Please check and try again."
# # #   exit 1
# # # fi

# # # echo "✅ Using project directory: $project_path"

# # # # ------------------- REPLACE PLACEHOLDERS IN YAML FILES -------------------
# # # declare -a files=(
# # #   "data_preprocessing/docker/job.yaml"
# # #   "model_training/docker/job.yaml"
# # #   "model_optimization/docker/job.yaml"
# # #   "app/docker/deployment.yaml"
# # #   "run.sh"
# # # )

# # # echo "🔄 Updating Docker Hub username in all configuration files..."
# # # for file in "${files[@]}"; do
# # #   sed -i "s|thisisangelz|$docker_username|g" "$file"
# # # done

# # # echo "✅ Updated all files with Docker username: $docker_username"

# # # # ------------------- START MINIKUBE -------------------
# # # echo "🚀 Starting Minikube..."
# # # minikube start --memory 4096 --cpus 2
# # # eval $(minikube docker-env)

# # # # ------------------- CREATE DIRECTORIES IN MINIKUBE -------------------
# # # echo "📂 Creating directories in Minikube..."
# # # minikube ssh -- "sudo mkdir -p /mnt/data/datasets/raw_datasets && sudo mkdir -p /mnt/data/datasets/cleaned_datasets && sudo mkdir -p /mnt/data/saved_model"

# # # # ------------------- COPY DATASETS TO MINIKUBE -------------------
# # # echo "📥 Copying datasets to Minikube..."
# # # minikube cp "$project_path/datasets/raw_datasets/train.csv" /mnt/data/datasets/raw_datasets/train.csv

# # # # ------------------- SET FILE PERMISSIONS -------------------
# # # echo "🔧 Setting file permissions..."
# # # minikube ssh -- "sudo chmod -R 777 /mnt/data/datasets/raw_datasets && sudo chmod -R 777 /mnt/data/datasets/cleaned_datasets && sudo chmod -R 777 /mnt/data/saved_model"

# # # # ------------------- PROMPT: PULL IMAGES FROM DOCKER HUB -------------------
# # # read -p "Do you want to pull existing images from Docker Hub? (y/n): " pull_choice

# # # if [[ $pull_choice == "y" ]]; then
# # #   echo "📥 Pulling Docker images from Docker Hub..."
# # #   docker pull $docker_username/data-preprocessing:latest
# # #   docker pull $docker_username/model-training:latest
# # #   docker pull $docker_username/model-optimization:latest
# # #   docker pull $docker_username/app:latest
# # # else
# # #   echo "⚠️ Skipping image pull. Proceeding with local builds..."
# # # fi

# # # # ------------------- BUILD DOCKER IMAGES -------------------
# # # echo "🐳 Building Docker images..."
# # # docker build -t $docker_username/data-preprocessing:latest -f data_preprocessing/docker/Dockerfile .
# # # docker build -t $docker_username/model-training:latest -f model_training/docker/Dockerfile .
# # # docker build -t $docker_username/model-optimization:latest -f model_optimization/docker/Dockerfile .
# # # docker build -t $docker_username/app:latest -f app/docker/Dockerfile .

# # # # ------------------- PROMPT: PUSH IMAGES TO DOCKER HUB -------------------
# # # read -p "Do you want to push the newly built images to Docker Hub? (y/n): " push_choice

# # # if [[ $push_choice == "y" ]]; then
# # #   echo "📤 Pushing Docker images to Docker Hub..."
# # #   docker push $docker_username/data-preprocessing:latest
# # #   docker push $docker_username/model-training:latest
# # #   docker push $docker_username/model-optimization:latest
# # #   docker push $docker_username/app:latest
# # # else
# # #   echo "⚠️ Skipping image push."
# # # fi

# # # # ------------------- APPLY KUBERNETES CONFIGURATIONS -------------------
# # # echo "⚙️ Applying Kubernetes configurations..."
# # # kubectl apply -f storage/pv_raw.yaml
# # # kubectl apply -f storage/pvc_raw.yaml
# # # kubectl apply -f storage/pv_cleaned.yaml
# # # kubectl apply -f storage/pvc_cleaned.yaml
# # # kubectl apply -f storage/pv_model.yaml
# # # kubectl apply -f storage/pvc_model.yaml

# # # # ------------------- DEPLOY CONTAINERS IN KUBERNETES -------------------
# # # kubectl apply -f data_preprocessing/docker/job.yaml
# # # kubectl apply -f model_training/docker/job.yaml
# # # kubectl apply -f model_optimization/docker/job.yaml
# # # kubectl apply -f app/docker/deployment.yaml
# # # kubectl apply -f app/docker/service.yaml

# # # # ------------------- DISPLAY JOB OUTPUTS (LOGS) -------------------
# # # echo "📜 Fetching logs for data preprocessing..."
# # # preprocessing_pod=$(kubectl get pods --selector=app=data-preprocessing -o jsonpath="{.items[0].metadata.name}")
# # # kubectl logs $preprocessing_pod

# # # echo "📜 Fetching logs for model training..."
# # # training_pod=$(kubectl get pods --selector=app=model-training -o jsonpath="{.items[0].metadata.name}")
# # # kubectl logs $training_pod

# # # echo "📜 Fetching logs for model optimization..."
# # # optimization_pod=$(kubectl get pods --selector=app=model-optimization -o jsonpath="{.items[0].metadata.name}")
# # # kubectl logs $optimization_pod

# # # # ------------------- OPTIONAL: SCALING & ROLLBACK -------------------
# # # read -p "Do you want to perform scaling or rollout operations now? (y/n): " scale_choice
# # # if [[ $scale_choice == "y" ]]; then
# # #   echo "📊 Adding scaling, rollout, and rollback capabilities..."
# # #   kubectl scale deployment streamlit-app --replicas=3
# # #   kubectl rollout status deployment streamlit-app
# # #   kubectl rollout history deployment streamlit-app
# # #   kubectl rollout undo deployment streamlit-app
# # # else
# # #   echo "⏩ Skipping scaling and rollout operations."
# # # fi

# # # # ------------------- CHECK POD & SERVICE STATUS -------------------
# # # echo "📡 Checking Kubernetes resources..."
# # # kubectl get pods
# # # kubectl get services

# # # # ------------------- WAIT FOR STREAMLIT APP TO BE READY -------------------
# # # echo "⏳ Waiting for Streamlit app to be ready..."
# # # kubectl wait --for=condition=ready pod --selector=app=streamlit-app --timeout=120s

# # # # ------------------- PORT FORWARDING TO ACCESS STREAMLIT APP -------------------
# # # echo "🔗 Port forwarding Streamlit app to localhost:8501..."
# # # kubectl port-forward service/streamlit-service 8501:8501 &

# # # echo "✅ Deployment completed! Open the app at: http://127.0.0.1:8501"










# # # update the push/pull logic
# # #!/bin/bash

# # # ------------------- USER INPUT: DOCKER HUB USERNAME -------------------
# # read -p "Enter your Docker Hub username: " docker_username

# # # Validate Docker username input
# # if [[ -z "$docker_username" ]]; then
# #   echo "❌ Error: Docker username cannot be empty!"
# #   exit 1
# # fi

# # # ------------------- USER INPUT: PROJECT DIRECTORY PATH -------------------
# # read -p "Enter the full path where the project folder (EGT309_Project_TeamMAL-main) is located: " project_path

# # # Validate project path input
# # if [[ ! -d "$project_path" ]]; then
# #   echo "❌ Error: The directory '$project_path' does not exist. Please check and try again."
# #   exit 1
# # fi

# # echo "✅ Using project directory: $project_path"

# # # ------------------- REPLACE PLACEHOLDERS IN YAML FILES -------------------
# # declare -a files=(
# #   "data_preprocessing/docker/job.yaml"
# #   "model_training/docker/job.yaml"
# #   "model_optimization/docker/job.yaml"
# #   "app/docker/deployment.yaml"
# #   "run.sh"
# # )

# # echo "🔄 Updating Docker Hub username in all configuration files..."
# # for file in "${files[@]}"; do
# #   sed -i "s|thisisangelz|$docker_username|g" "$file"
# # done

# # echo "✅ Updated all files with Docker username: $docker_username"

# # # ------------------- START MINIKUBE -------------------
# # echo "🚀 Starting Minikube..."
# # minikube start --memory 4096 --cpus 2
# # eval $(minikube docker-env)

# # # ------------------- CREATE DIRECTORIES IN MINIKUBE -------------------
# # echo "📂 Creating directories in Minikube..."
# # minikube ssh -- "sudo mkdir -p /mnt/data/datasets/raw_datasets && sudo mkdir -p /mnt/data/datasets/cleaned_datasets && sudo mkdir -p /mnt/data/saved_model"

# # # ------------------- COPY DATASETS TO MINIKUBE -------------------
# # echo "📥 Copying datasets to Minikube..."
# # minikube cp "$project_path/datasets/raw_datasets/train.csv" /mnt/data/datasets/raw_datasets/train.csv

# # # ------------------- SET FILE PERMISSIONS -------------------
# # echo "🔧 Setting file permissions..."
# # minikube ssh -- "sudo chmod -R 777 /mnt/data/datasets/raw_datasets && sudo chmod -R 777 /mnt/data/datasets/cleaned_datasets && sudo chmod -R 777 /mnt/data/saved_model"

# # # ------------------- PROMPT: PULL IMAGES FROM DOCKER HUB -------------------
# # read -p "Do you want to pull existing images from Docker Hub? (y/n): " pull_choice
# # build_images=true  # Default to true, will be set to false if pulling succeeds

# # if [[ $pull_choice == "y" ]]; then
# #   echo "📥 Pulling Docker images from Docker Hub..."
# #   docker pull $docker_username/data-preprocessing:latest &&
# #   docker pull $docker_username/model-training:latest &&
# #   docker pull $docker_username/model-optimization:latest &&
# #   docker pull $docker_username/app:latest && build_images=false

# #   if [[ $build_images == false ]]; then
# #     echo "✅ Images pulled successfully. Skipping build process."
# #   else
# #     echo "⚠️ One or more images failed to pull. Proceeding with local build..."
# #   fi
# # else
# #   echo "⚠️ Skipping image pull. Proceeding with local builds..."
# # fi

# # # ------------------- BUILD DOCKER IMAGES IF NOT PULLED -------------------
# # if [[ $build_images == true ]]; then
# #   echo "🐳 Building Docker images..."
# #   docker build -t $docker_username/data-preprocessing:latest -f data_preprocessing/docker/Dockerfile .
# #   docker build -t $docker_username/model-training:latest -f model_training/docker/Dockerfile .
# #   docker build -t $docker_username/model-optimization:latest -f model_optimization/docker/Dockerfile .
# #   docker build -t $docker_username/app:latest -f app/docker/Dockerfile .
# # fi

# # # ------------------- PROMPT: PUSH IMAGES TO DOCKER HUB -------------------
# # read -p "Do you want to push the newly built images to Docker Hub? (y/n): " push_choice

# # if [[ $push_choice == "y" ]]; then
# #   echo "📤 Pushing Docker images to Docker Hub..."
# #   docker push $docker_username/data-preprocessing:latest
# #   docker push $docker_username/model-training:latest
# #   docker push $docker_username/model-optimization:latest
# #   docker push $docker_username/app:latest
# # else
# #   echo "⚠️ Skipping image push."
# # fi

# # # ------------------- APPLY KUBERNETES CONFIGURATIONS -------------------
# # echo "⚙️ Applying Kubernetes configurations..."
# # kubectl apply -f storage/pv_raw.yaml
# # kubectl apply -f storage/pvc_raw.yaml
# # kubectl apply -f storage/pv_cleaned.yaml
# # kubectl apply -f storage/pvc_cleaned.yaml
# # kubectl apply -f storage/pv_model.yaml
# # kubectl apply -f storage/pvc_model.yaml

# # # ------------------- DEPLOY CONTAINERS IN KUBERNETES -------------------
# # kubectl apply -f data_preprocessing/docker/job.yaml
# # kubectl apply -f model_training/docker/job.yaml
# # kubectl apply -f model_optimization/docker/job.yaml
# # kubectl apply -f app/docker/deployment.yaml
# # kubectl apply -f app/docker/service.yaml

# # # ------------------- DISPLAY JOB OUTPUTS (LOGS) -------------------
# # echo "📜 Fetching logs for data preprocessing..."
# # preprocessing_pod=$(kubectl get pods --selector=app=data-preprocessing -o jsonpath="{.items[0].metadata.name}")
# # kubectl logs $preprocessing_pod

# # echo "📜 Fetching logs for model training..."
# # training_pod=$(kubectl get pods --selector=app=model-training -o jsonpath="{.items[0].metadata.name}")
# # kubectl logs $training_pod

# # echo "📜 Fetching logs for model optimization..."
# # optimization_pod=$(kubectl get pods --selector=app=model-optimization -o jsonpath="{.items[0].metadata.name}")
# # kubectl logs $optimization_pod

# # # ------------------- OPTIONAL: SCALING & ROLLBACK -------------------
# # read -p "Do you want to perform scaling or rollout operations now? (y/n): " scale_choice
# # if [[ $scale_choice == "y" ]]; then
# #   echo "📊 Adding scaling, rollout, and rollback capabilities..."
# #   kubectl scale deployment streamlit-app --replicas=3
# #   kubectl rollout status deployment streamlit-app
# #   kubectl rollout history deployment streamlit-app
# #   kubectl rollout undo deployment streamlit-app
# # else
# #   echo "⏩ Skipping scaling and rollout operations."
# # fi

# # # ------------------- CHECK POD & SERVICE STATUS -------------------
# # echo "📡 Checking Kubernetes resources..."
# # kubectl get pods
# # kubectl get services

# # # ------------------- WAIT FOR STREAMLIT APP TO BE READY -------------------
# # echo "⏳ Waiting for Streamlit app to be ready..."
# # kubectl wait --for=condition=ready pod --selector=app=streamlit-app --timeout=120s

# # # ------------------- PORT FORWARDING TO ACCESS STREAMLIT APP -------------------
# # echo "🔗 Port forwarding Streamlit app to localhost:8501..."
# # kubectl port-forward service/streamlit-service 8501:8501 &

# # echo "✅ Deployment completed! Open the app at: http://127.0.0.1:8501"










# # update display logs

# #!/bin/bash

# # Ensure the script has correct permissions
# if [[ ! -x "$0" ]]; then
#   echo "🔧 Setting executable permissions for run.sh..."
#   chmod +x "$0"
# fi

# # ------------------- USER INPUT: DOCKER HUB USERNAME -------------------
# read -p "Enter your Docker Hub username: " docker_username

# # Validate Docker username input
# if [[ -z "$docker_username" ]]; then
#   echo "❌ Error: Docker username cannot be empty!"
#   exit 1
# fi

# # ------------------- USER INPUT: PROJECT DIRECTORY PATH -------------------
# read -p "Enter the full path where the project folder (EGT309_Project_TeamMAL-main) is located: " project_path

# # Validate project path input
# if [[ ! -d "$project_path" ]]; then
#   echo "❌ Error: The directory '$project_path' does not exist. Please check and try again."
#   exit 1
# fi

# echo "✅ Using project directory: $project_path"

# # ------------------- REPLACE PLACEHOLDERS IN YAML FILES -------------------
# declare -a files=(
#   "data_preprocessing/docker/job.yaml"
#   "model_training/docker/job.yaml"
#   "model_optimization/docker/job.yaml"
#   "app/docker/deployment.yaml"
#   "run.sh"
# )

# echo "🔄 Updating Docker Hub username in all configuration files..."
# for file in "${files[@]}"; do
#   sed -i "s|thisisangelz|$docker_username|g" "$file"
# done

# echo "✅ Updated all files with Docker username: $docker_username"

# # ------------------- START MINIKUBE -------------------
# echo "🚀 Starting Minikube..."
# minikube start --memory 4096 --cpus 2
# eval $(minikube docker-env)

# # ------------------- CREATE DIRECTORIES IN MINIKUBE -------------------
# echo "📂 Creating directories in Minikube..."
# minikube ssh -- "sudo mkdir -p /mnt/data/datasets/raw_datasets && sudo mkdir -p /mnt/data/datasets/cleaned_datasets && sudo mkdir -p /mnt/data/saved_model"

# # ------------------- COPY DATASETS TO MINIKUBE -------------------
# echo "📥 Copying datasets to Minikube..."
# minikube cp "$project_path/datasets/raw_datasets/train.csv" /mnt/data/datasets/raw_datasets/train.csv

# # ------------------- SET FILE PERMISSIONS -------------------
# echo "🔧 Setting file permissions..."
# minikube ssh -- "sudo chmod -R 777 /mnt/data/datasets/raw_datasets && sudo chmod -R 777 /mnt/data/datasets/cleaned_datasets && sudo chmod -R 777 /mnt/data/saved_model"

# # ------------------- PROMPT: PULL IMAGES FROM DOCKER HUB -------------------
# read -p "Do you want to pull existing images from Docker Hub? (y/n): " pull_choice
# build_images=true  # Default to true, will be set to false if pulling succeeds

# if [[ $pull_choice == "y" ]]; then
#   echo "📥 Pulling Docker images from Docker Hub..."
#   docker pull $docker_username/data-preprocessing:latest &&
#   docker pull $docker_username/model-training:latest &&
#   docker pull $docker_username/model-optimization:latest &&
#   docker pull $docker_username/app:latest && build_images=false

#   if [[ $build_images == false ]]; then
#     echo "✅ Images pulled successfully. Skipping build process."
#   else
#     echo "⚠️ One or more images failed to pull. Proceeding with local build..."
#   fi
# else
#   echo "⚠️ Skipping image pull. Proceeding with local builds..."
# fi

# # ------------------- BUILD DOCKER IMAGES IF NOT PULLED -------------------
# if [[ $build_images == true ]]; then
#   echo "🐳 Building Docker images..."
#   docker build -t $docker_username/data-preprocessing:latest -f data_preprocessing/docker/Dockerfile .
#   docker build -t $docker_username/model-training:latest -f model_training/docker/Dockerfile .
#   docker build -t $docker_username/model-optimization:latest -f model_optimization/docker/Dockerfile .
#   docker build -t $docker_username/app:latest -f app/docker/Dockerfile .
# fi

# # ------------------- PROMPT: PUSH IMAGES TO DOCKER HUB -------------------
# read -p "Do you want to push the newly built images to Docker Hub? (y/n): " push_choice

# if [[ $push_choice == "y" ]]; then
#   echo "📤 Pushing Docker images to Docker Hub..."
#   docker push $docker_username/data-preprocessing:latest
#   docker push $docker_username/model-training:latest
#   docker push $docker_username/model-optimization:latest
#   docker push $docker_username/app:latest
# else
#   echo "⚠️ Skipping image push."
# fi

# # ------------------- APPLY KUBERNETES CONFIGURATIONS -------------------
# echo "⚙️ Applying Kubernetes configurations..."
# kubectl apply -f storage/pv_raw.yaml
# kubectl apply -f storage/pvc_raw.yaml
# kubectl apply -f storage/pv_cleaned.yaml
# kubectl apply -f storage/pvc_cleaned.yaml
# kubectl apply -f storage/pv_model.yaml
# kubectl apply -f storage/pvc_model.yaml

# # ------------------- DEPLOY CONTAINERS IN KUBERNETES -------------------
# kubectl apply -f data_preprocessing/docker/job.yaml
# kubectl apply -f model_training/docker/job.yaml
# kubectl apply -f model_optimization/docker/job.yaml
# kubectl apply -f app/docker/deployment.yaml
# kubectl apply -f app/docker/service.yaml

# # ------------------- DISPLAY JOB OUTPUTS (LOGS) -------------------
# echo "📜 Fetching logs for data preprocessing..."
# preprocessing_pod=$(kubectl get pods --selector=app=data-preprocessing -o jsonpath="{.items[0].metadata.name}")
# kubectl logs $preprocessing_pod

# echo "📜 Fetching logs for model training..."
# training_pod=$(kubectl get pods --selector=app=model-training -o jsonpath="{.items[0].metadata.name}")
# kubectl logs $training_pod

# echo "📜 Fetching logs for model optimization..."
# optimization_pod=$(kubectl get pods --selector=app=model-optimization -o jsonpath="{.items[0].metadata.name}")
# kubectl logs $optimization_pod

# # ------------------- OPTIONAL: SCALING & ROLLBACK -------------------
# read -p "Do you want to perform scaling or rollout operations now? (y/n): " scale_choice
# if [[ $scale_choice == "y" ]]; then
#   echo "📊 Adding scaling, rollout, and rollback capabilities..."
#   kubectl scale deployment streamlit-app --replicas=3
#   kubectl rollout status deployment streamlit-app
#   kubectl rollout history deployment streamlit-app
#   kubectl rollout undo deployment streamlit-app
# else
#   echo "⏩ Skipping scaling and rollout operations."
# fi

# # ------------------- CHECK POD & SERVICE STATUS -------------------
# echo "📡 Checking Kubernetes resources..."
# kubectl get pods
# kubectl get services

# # ------------------- WAIT FOR STREAMLIT APP TO BE REA













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
read -p "Enter the full path where the project folder (EGT309_Project_TeamMAL-main) is located: " project_path

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
  sed -i "s|thisisangelz|$docker_username|g" "$file"
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

# ------------------- OPTIONAL: SCALING & ROLLBACK -------------------
read -p "Do you want to perform scaling or rollout operations now? (y/n): " scale_choice
if [[ $scale_choice == "y" ]]; then
  echo "📊 Adding scaling, rollout, and rollback capabilities..."
  kubectl scale deployment streamlit-app --replicas=3
  kubectl rollout status deployment streamlit-app
  kubectl rollout history deployment streamlit-app
  kubectl rollout undo deployment streamlit-app
else
  echo "⏩ Skipping scaling and rollout operations."
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
