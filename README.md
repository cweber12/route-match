_______________________________________________________
##### CONTENTS #####
_______________________________________________________

# app_match

Docker container: route-map-app
Light backend functions for: 
- login/auth
- feature matching
- retrieving data from s3 

______________________________________________________
##### CLI COMMANDS #####
______________________________________________________

# DOCKER

BUILD:                docker build -t route-map-match .

STOP:                 docker stop route-map-match

REMOVE:               docker rm route-map-match

RUN:                  docker run --env-file .env -p 8000:8000 --name route-map-match route-map-match

LIST RUNNING:         docker ps

CLEAR CACHE:          docker system prune

________________________________________________________

# server.py

import os
from app_process.main import app
import uvicorn
import ultralytics


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)

_______________________________________________________

# Dockerfile (process)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app_process

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements_process.txt .
RUN pip install --upgrade pip && pip install -r requirements_process.txt

COPY app_process/ .
COPY pose_landmarker_full.task .
COPY pose_landmarker_heavy.task .
COPY pose_landmarker_lite.task .
COPY yolov5s.pt .
COPY yolov8n.pt .

RUN mkdir -p /app_process/temp_uploads

ENV AWS_DEFAULT_REGION=us-east-2

EXPOSE 80

CMD ["uvicorn", "app_process.main:app", "--host", "0.0.0.0", "--port", "80"]

_________________________________________________________________________-

# ECR

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 537124934274.dkr.ecr.us-east-2.amazonaws.com

docker tag route-map-match:latest 537124934274.dkr.ecr.us-east-2.amazonaws.com/route-map-match:latest
      
docker push 537124934274.dkr.ecr.us-east-2.amazonaws.com/route-map-match:latest

# EC2 SSH

sudo docker pull 537124934274.dkr.ecr.us-east-2.amazonaws.com/route-map-match:latest

aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin 537124934274.dkr.ecr.us-east-2.amazonaws.com

sudo docker pull 537124934274.dkr.ecr.us-east-2.amazonaws.com/route-map-match:latest

