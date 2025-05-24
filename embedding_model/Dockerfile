# Chose Docker Image
FROM python:3.12-slim

# Set working directory
WORKDIR /usr/src/app

# Install dependecies for the application
RUN apt-get update && \
    apt-get upgrade && \
    apt-get install -y --no-install-recommends build-essential python3-dev gfortran && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY ./requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the main code
COPY ./embedding_model.py .

# Expose the port for connection to the application
EXPOSE 8000

# Run the server
CMD ["python", "./embedding_model.py"]
