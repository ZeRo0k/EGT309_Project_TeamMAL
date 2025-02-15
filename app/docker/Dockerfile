# Use the official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Copy the requirements file and install dependencies
RUN pip install --no-cache-dir -r /app/python_scripts/requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
ENTRYPOINT ["streamlit", "run", "/app/python_scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
