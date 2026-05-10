# Use a lightweight python base image
FROM python:3.10-slim

# Install necessary system dependencies (if any are needed by torch/transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*



# # Use a lightweight python base image
# FROM python:3.10-slim
# 
# # Install necessary system dependencies (if any are needed by torch/transformers)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*
# 
