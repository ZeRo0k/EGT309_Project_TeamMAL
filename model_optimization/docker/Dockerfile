# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/python_scripts/requirements.txt

# Entry point (modify as needed)
CMD ["python", "python_scripts/model_optimization.py"]