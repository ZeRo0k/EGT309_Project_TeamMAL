# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the relevant directories and files
COPY ./model_optimization/python_scripts /app/python_scripts
COPY ./model_optimization/docker /app/docker
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Entry point (modify as needed)
CMD ["python", "/app/python_scripts/model_optimization.py"]
