# This file defines the containerized environment for running the Titanic 
# survival prediction app using Streamlit.
#
# Key Features:
# - Uses Python 3.10 Slim as a lightweight base image.
# - Installs required dependencies.
# - Copies preprocessing scripts to ensure feature consistency.
# - Runs Streamlit on port 8501.

# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the relevant directories and files
COPY ./app/python_scripts /app/python_scripts
COPY ./app/docker /app/docker
COPY ./requirements.txt /app/requirements.txt
COPY ./data_preprocessing /app/data_preprocessing

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "/app/python_scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]