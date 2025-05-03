# docker file to build and run solution.py

#base image
FROM python:3.11-slim

#files to copy
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
libglib2.0-0 libsm6 libxrender1 libxext6 \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#command to run the training job
CMD ["python", "solution.py", "--mode", "test", "--feature_type", "hog", "--num_unit", "64"]
