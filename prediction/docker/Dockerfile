# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy all project files into the container
COPY ./prediction /app

# Copy the shared requirements.txt from the parent directory
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the default command for the container
CMD ["python3", "python_scripts/prediction.py"]
