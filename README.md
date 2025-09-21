
# RouteMap Backend Match

Backend for climbing route comparison, pose estimation, and video generation. Provides FastAPI endpoints for image upload, SIFT/pose matching, job management, and browser-ready video output. Includes Docker and AWS deployment instructions, and tools for route export and architecture diagram generation.

---

## CLI Commands

### Docker (Local)

#### Stop and Remove

```bash
docker stop route-map-match

docker rm route-map-match
```

#### BUILD AND RUN

```bash
docker build -t route-map-match .

docker run --env-file .env -p 8000:8000 --name route-map-match route-map-match
```

### AWS Elastic Container Registry (ECR)

#### Login

``` bash
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 537124934274.dkr.ecr.us-east-2.amazonaws.com
```

#### TAG IMAGE TO PUSH

``` bash
docker tag route-map-match:latest 537124934274.dkr.ecr.us-east-2.amazonaws.com/route-map-match:latest
```

#### PUSH TAGGED IMAGE TO ECR

``` bash
docker push 537124934274.dkr.ecr.us-east-2.amazonaws.com/route-map-match:latest
```

### AWS Elastic Compute (EC2)

#### Open Secure Shell to EC2 Instance

```bash
ssh -i "C:\Projects\ec2-key\rm-key.pem" ec2-user@ec2-3-22-80-166.us-east-2.compute.amazonaws.com
```

#### VERIFY ECR CREDENTIALS BEFORE PULL

```bash
aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin 537124934274.dkr.ecr.us-east-2.amazonaws.com
```

#### PULL LATEST IMAGE FROM ECR

```bash
sudo docker pull 537124934274.dkr.ecr.us-east-2.amazonaws.com/route-map-match:latest
```

#### STOP AND REMOVE CURRENT CONTAINER RUNNING ON EC2

```bash
docker stop route-map-match || true
docker rm route-map-match || true
```

#### RUN DOCKER CONTAINER IN THE BACKGROUND

```bash
sudo docker run -d --env-file .env -p 8000:8000 --name route-map-match 537124934274.dkr.ecr.us-east-2.amazonaws.com/route-map-match:latest
```

## Architecture Diagram & Route Export

### Export Routes

```bash
$env:ROUTE_EXPORT = "1"  
```

```bash
$env:APP_MODULE   = "app.main:app"
```

```bash
python tools/export_routes.py --out routes.json
```

### Generate Handler

```bash
python tools/callgraph_ast.py app --out callgraph.json --prefix app
