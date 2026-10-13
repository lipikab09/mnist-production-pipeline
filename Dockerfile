# Use a lightweight official Python image
FROM python:3.14-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first (for efficient Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and model weights
COPY app/ ./app/
COPY models/ ./models/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]