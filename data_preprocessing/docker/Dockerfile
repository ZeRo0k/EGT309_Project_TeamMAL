# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the relevant directories and files
COPY ./data_preprocessing/python_scripts /app/python_scripts
COPY ./data_preprocessing/docker /app/docker
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Entry point (modify as needed)
CMD ["python", "/app/python_scripts/data_preprocessing.py"]