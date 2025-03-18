# Use a Python base image
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (default for FastAPI)
EXPOSE 8000

# Command to start the server
# Cloud Run expects the service to listen on 0.0.0.0 and the port is 8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]