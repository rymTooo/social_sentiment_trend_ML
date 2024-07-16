#!/bin/sh

# Configure MinIO client
# mc alias set myminio http://0.0.0.0:29000 admin admin123

# Check if bucket exists
if mc ls myminio/mlflow > /dev/null 2>&1; then
    echo "Bucket mlflow already exists"
else
    echo "Creating bucket mlflow"
    mc mb myminio/mlflow
fi

# Start MinIO server
exec "$@"