# Use a lightweight python base image
FROM python:3.10-slim

# Install necessary system dependencies (if any are needed by torch/transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user for HuggingFace Spaces
RUN useradd -m -u 1000 user


# # Use a lightweight python base image
# FROM python:3.10-slim
# 
# # Install necessary system dependencies (if any are needed by torch/transformers)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*
# 
# # Set working directory
# WORKDIR /code
# 
# # Copy requirements and install
# COPY requirements.txt /code/requirements.txt
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# 
# # Create a non-root user for HuggingFace Spaces
# RUN useradd -m -u 1000 user
