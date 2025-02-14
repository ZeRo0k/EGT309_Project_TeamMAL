# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy files
COPY . /app

# Copy the requirements file and install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the saved model into the container
COPY ../saved_model /app/saved_model

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
