# Use the base Python image with version specifier
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for MLflow tracking and model serving
EXPOSE 5000 8000

# Copy your application code into the container
COPY . .

# Set the entrypoint command to run the MLflow tracking server and model serving
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/app/mlruns", "--workers", "1", "&", "python", "service/main.py", "--port", "8000"]