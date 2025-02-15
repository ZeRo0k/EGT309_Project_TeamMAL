# Use the official Python slim image for a smaller build
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy application files (except saved_model)
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r python_scripts/requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "python_scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
