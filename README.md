# **End-to-End ML Deployment: Titanic Survival Prediction with Kubernetes & Docker** 

## **1. Project Overview**

### **Project Title:**
**End-to-End ML Deployment: Titanic Survival Prediction with Kubernetes & Docker**
####
#### **Team Name**: **MAL**

###
#### **Objective:**
📌 **To build a scalable, automated, and containerized ML pipeline** that streamlines **data processing, model deployment, and real-time predictions** in a **Kubernetes environment**.
###
#### **Description:**
This project develops a **containerized ML pipeline** to predict **Titanic passenger survival** using a web application. The pipeline consists of:

- **Data Preprocessing** – Cleans and transforms raw passenger data for model training.
- **Model Training & Optimization** – Builds and fine-tunes an ML model for survival prediction.
- **Web-Based UI (Streamlit)** – Provides a user-friendly interface for **Single & Batch Predictions**.
- **Containerization & Orchestration** – Managed using **Docker & Kubernetes (Minikube)** for deployment.

---

## **2. System Architecture**


#### **Folder Structure**
```
EGT309_Project_TeamMAL-main/
│── app/
│   ├── docker/
│   │   ├── deployment.yaml
│   │   ├── Dockerfile
│   │   ├── service.yaml
│   ├── python_scripts/
│   │   ├── app.py
│   │   ├── config.yaml
│
│── data_preprocessing/
│   ├── docker/
│   │   ├── deployment.yaml
│   │   ├── Dockerfile
│   │   ├── job.yaml
│   ├── python_scripts/
│   │   ├── __init__.py
│   │   ├── config.yaml
│   │   ├── data_preprocessing.py
│
│── datasets/
│   ├── cleaned_datasets/
│   │   ├── cleaned_train_data.csv
│   ├── raw_datasets/
│   │   ├── test.csv
│   │   ├── train.csv
│
│── model_optimization/
│   ├── docker/
│   │   ├── deployment.yaml
│   │   ├── Dockerfile
│   │   ├── job.yaml
│   ├── python_scripts/
│   │   ├── config.yaml
│   │   ├── model_optimization.py
│
│── model_training/
│   ├── docker/
│   │   ├── deployment.yaml
│   │   ├── Dockerfile
│   │   ├── job.yaml
│   ├── python_scripts/
│   │   ├── config.yaml
│   │   ├── model_training.py
│
│── storage/
│   ├── pv_cleaned.yaml
│   ├── pv_model.yaml
│   ├── pv_raw.yaml
│   ├── pvc_cleaned.yaml
│   ├── pvc_model.yaml
│   ├── pvc_raw.yaml
│
│── .gitignore
│── README.md
│── requirements.txt
│── run.sh
│── EGT309_Project_TeamMAL_PPT.pptx
│── System Architecture Diagram.png
```

###

#### **Containers Used:**
- **Data Preprocessing**
- **Model Training**
- **Model Optimization**
- **Model Inference (App)**
###
#### **Technologies Used:**
- **Development Environment:** Visual Studio Code  
- **Containerization & Orchestration:** Docker, Kubernetes, Minikube  
- **Web & UI Framework:** Streamlit  
- **Programming Language:** Python  
- **Libraries & Frameworks:** Pandas, scikit-learn, Joblib, PyYAML  



###
#### **High-Level Architecture Diagram**
![alt text](<System Architecture Diagram.png>)

 
**Overview of system architecture diagram:**
This section provides a comprehensive overview of the Kubernetes system architecture, detailing the key components and workflow involved in data processing, model training, and application deployment. The architecture is designed to ensure efficient resource management, scalability, and high availability within a Minikube environment.

The system is initialized on a local machine, where the developer creates and manages a Kubernetes cluster using Minikube. The setup process involves configuring Persistent Volumes (PVs) and Persistent Volume Claims (PVCs) to store different types of data across various processing stages.

Initially, a raw data PV is created to store unprocessed data uploaded from the local machine. The data preprocessing job accesses the raw data PV through a PVC, retrieving the data for cleaning and transformation. Once preprocessing is completed, the cleaned data is stored in a cleaned data PV, ensuring that subsequent processes operate on refined data.

The model training and optimization jobs then access the cleaned data PV via the cleaned data PVC. These jobs train machine learning models using different hyperparameters. To optimize storage efficiency, only the best-performing model from the model optimization process is retained in a model PV, ensuring that redundant models do not consume unnecessary storage.

Following model training and selection, the application deployment component retrieves the best-trained model from the model PV via the model PVC. The retrieved model is then used within the deployed application, which runs on Kubernetes pods with replicas to enhance availability and reliability.

On the user side, user requests are sent to a load balancer, which manages incoming traffic by directing it to the appropriate service. The Kubernetes service then exposes the application’s port externally, allowing users to interact with the deployed model through a web server interface. 

This architecture ensures seamless integration between data processing, model training, and real-time application access while maintaining scalability and fault tolerance. This structured approach facilitates efficient resource utilization, supports high-availability applications, and enables automated model deployment in a Kubernetes-based environment.

---

## **3. Project Components** 

#### **a. Data Preprocessing** (`data_preprocessing` folder)
Processes **raw Titanic dataset** by handling **missing values**, **encoding categorical features**, performing **feature engineering**, **removing outliers**, and **dropping unnecessary features**.  
The preprocessing job runs in a **Docker container** orchestrated by **Kubernetes Jobs**, storing both **raw and cleaned data** in **Persistent Volume Claims (PVCs)**.



#### **b. Model Training** (`model_training` folder)
Trains an **ML model** using the **cleaned dataset**, applying **data splitting, training, and evaluation**.  
The training job is **containerized** and managed via **Kubernetes Jobs**, ensuring **scalability and reproducibility**.



#### **c. Model Optimization** (`model_optimization` folder)
Fine-tunes the trained model using **hyperparameter tuning techniques**.  
The optimized model is **stored in PVCs** and serves as the **final model for predictions**.  
The optimization process runs in a **Dockerized Kubernetes Job**, making it **efficient and repeatable**.



#### **d. Datasets** (`datasets` folder)
Stores both **raw** and **cleaned** datasets, ensuring smooth **data flow between preprocessing, training, and optimization**.  
The datasets are stored in **PVCs** to maintain **persistence** across different jobs.



#### **e. Storage** (`storage` folder)
Manages **Persistent Volumes (PVs)** and **Persistent Volume Claims (PVCs)** to ensure data, models, and intermediate results **remain available** across different stages of the pipeline.



#### **f. Requirements to be Installed** (`requirements.txt`)
Lists all the required dependencies (**Python libraries**) necessary for the **ML pipeline, model training, and deployment**.



#### **g. Bash Script** (`run.sh`)
Automates the **entire deployment process**, including:
- 🚀 **Minikube initialization**
- 🐳 **Building and pushing Docker images**
- ⚙️ **Deploying Kubernetes Jobs and Services**
- 📂 **Setting up PVCs for data persistence**
- 🔄 **Running the ML pipeline end-to-end**

Running `run.sh` sets up the **entire ML pipeline automatically** for deployment. 🚀

---
## 4. Docker & Kubernetes

### Introduction to Docker
This project utilizes **Docker** to containerize applications, ensuring **portability, scalability**, and consistency across different environments. By using Docker, we eliminate **compatibility issues**, simplify **dependency management**, and streamline deployment across infrastructures.

###

### Key Docker Components in This Project:

#### 🛠️ Dockerfile
Each service runs inside a **dedicated container**, built using its respective **Dockerfile**. This file defines:
- ✅ **Base Image** (e.g., Python 3.10)
- 📦 **Dependencies** (installed via `requirements.txt`)
- ⚙️ **Runtime Configurations** (e.g., exposed ports)
- 🌍 **Environment Variables**

Each ML pipeline component has its own **Dockerfile**, ensuring **modularity, consistency, and reproducibility**.

#### 📦 Containerized Python Scripts
The following Python scripts are **containerized** and executed inside **Docker containers**:
- `data_preprocessing.py` → Cleans and transforms raw data.
- `model_training.py` → Trains the machine learning model.
- `model_optimization.py` → Fine-tunes and selects the best model.
- `app.py` → Runs the **Streamlit Web UI** for real-time predictions.

These scripts **work seamlessly together** within a **containerized ML pipeline**, ensuring **portability and efficiency**.

##
### Docker Workflow:
🚀 The container lifecycle follows these steps:
1. **Build** the Docker image (using `Dockerfile`).  
2. **Run** the container locally for testing.  
3. **Push** the image to a Container Registry (e.g., Docker Hub).  
4. **Deploy** the container in a **Kubernetes cluster**.



### Benefits of Using Docker:
- ✅ **Portability** – Runs consistently across development, testing, and production.
- ✅ **Scalability** – Works with **Kubernetes** for auto-scaling.
- ✅ **Efficiency** – Uses fewer resources than traditional VMs.
- ✅ **Fast Deployment** – Containers start quickly and are easy to update.

##
### Kubernetes Components:

#### 🔹 Pods
- The **smallest deployable unit** in Kubernetes.
- Each container runs inside a **dedicated pod**, ensuring **modularity** and **fault tolerance**.

#### 🚀 Deployments
- Manages **application instances**.
- Ensures **scalability, high availability, and rolling updates**.

#### ⏳ Jobs (One-Time Tasks)
Used for **batch processing** tasks that execute once and then stop:
- **Data Preprocessing** → Cleans and saves data for model training.
- **Model Training** → Evaluates the trained model.
- **Model Optimization** → Tunes hyperparameters and saves the best model.

##

### 📂 Persistent Volumes (PVs):
Kubernetes **Persistent Volumes (PVs)** store data **permanently** in **Minikube**, ensuring data persists even after a **system shutdown**. However, **data will be lost if Minikube is deleted**.

#### PVs Used in This Project:
- 📂 `pv-raw` → Stores **raw data** (`train.csv`, `test.csv`).
- 📂 `pv-cleaned` → Stores **cleaned and preprocessed data**.
- 📂 `pv-model` → Stores the **optimized machine learning model**.

##
### 🌐 Networking & Load Management:
- 📌 **Scaling** → Adjusts application instances **dynamically**.
- 🔄 **Self-Healing** → Restarts failed containers **automatically**.
- ⚖️ **Load Balancing** → Distributes traffic across multiple pods.
- ♻️ **Rollback/Rollout** → Enables **version control** for deployments.

#### ☁️ Kubernetes Service
A **Service** exposes the application deployment to:
- 🔹 **Internal Kubernetes Pods** → Enables communication **within the cluster**.
- 🌍 **External Users** → Allows access from **outside the cluster**.



---
## **5. Deployment Setup and Instructions** 

### **Pre-Requisites**
⚠️ **Ensure the following are installed before running the setup:**
- Windows Subsystem for Linux (**WSL**)
- **Ubuntu**
- **Docker Desktop** (*Log in before proceeding*)


### **Setup Steps:**

#### **1️⃣ Download Project Files**
- Clone the repository or **download the ZIP file** from GitHub (main branch) into your local `Downloads` folder.
###

#### **2️⃣ Open Command Prompt & Navigate to Project Folder**
```
> wsl
> cd
> cd <path to this project folder>
eg. cd /mnt/c/Users/john/Downloads/EGT309_Project_TeamMAL-main/EGT309_Project_TeamMAL-main
```
###
#### **3️⃣ Run the Deployment Script**
```
> ./run.sh
```
---

## 6. Version Control 

### GitHub Repository
🔗 https://github.com/ZeRo0k/EGT309_Project_TeamMAL.git

### Version Control Strategy
This project follows a **structured Git workflow** using **GitHub for version control**, ensuring **collaboration, code integrity, and efficient tracking**.

### Branching Structure
📌 **`main` branch** – Production-ready, stable, and tested code is merged here.  

📌 **Individual branches** (`angel-branch`, `limenhuai-branch`, `mock-branch`) – Each team member worked in their own branch to develop:
- Data Preprocessing
- Model Training
- Model Optimization

### Git Workflow Process
✅ **Branching Strategy** – Each member worked **independently** on their assigned task to avoid merge conflicts.  

✅ **Commits & Pull Requests (PRs)** – Frequent commits with **clear messages**, followed by a **pull request (PR)** for review.  

✅ **Code Reviews** – PRs were reviewed by other members before merging into **`main`** to ensure quality.  

✅ **Merge & Conflict Resolution** – Conflicts were handled using **interactive rebasing** or **manual resolution** before merging.  

✅ **Issue Tracking** – GitHub **Issues** were used for tracking bugs, improvements, and module-specific discussions.  

📂 **Additional Documentation:**  
A separate document contains **Git commands, outputs, and troubleshooting steps**, structured similarly to the **EGT309 Labs instructions**, serving as a technical reference.
