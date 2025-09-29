# RouteScan Backend Match

Backend for climbing route comparison, pose estimation, and video generation. Provides FastAPI endpoints for image upload, SIFT/pose matching, job management, and browser-ready video output. Includes Docker and AWS deployment instructions, and tools for route export and architecture diagram generation.

---

## How Video Processing Works

When a user submits a comparison job, the backend performs the following steps:

1. **Image Upload & Data Selection**
   - Accepts an uploaded image or selects a built-in reference image.
   - Loads pose and SIFT feature data from Amazon S3 for the selected climbing route.

2. **Pose & SIFT Matching**
   - For single-frame jobs, SIFT feature matching computes a single affine transformation to align the pose landmarks to the reference image.
   - For multi-frame jobs, SIFT matching is performed for each frame, generating a unique transformation matrix per frame for dynamic sequences.

3. **Pose Transformation & Interpolation**
   - Pose landmarks are transformed using the computed affine matrices.
   - Linear interpolation is used to create smooth transitions between frames.

4. **Video Frame Generation**
   - Each frame is rendered by drawing the transformed pose landmarks onto the reference image.
   - Frames are written to a video file using OpenCV, with color and style parameters set by the user.
   - Progress is tracked and reported for each frame processed.

5. **Output & Delivery**
   - The final video is saved in `temp_uploads/pose_feature_data/output_video/`.
   - A browser-ready version is generated for frontend playback.
   - Temporary files and intermediate results are cleaned up after processing.
   - Output videos and processed data can be uploaded to S3 for long-term storage and secure delivery.

This pipeline ensures accurate pose alignment, smooth animation, and efficient video generation for climbing route comparisons. All processing is performed asynchronously in background jobs, with real-time progress and results available via the API.

---

## Data Handling & Storage

This project manages climbing route, pose, and video data using a combination of local storage and Amazon S3. Key aspects include:

1. **User Route Data**
   - Route and area trees are cached in S3 to speed up browsing and reduce repeated data loads.
   - Recent attempts and route coordinates are fetched from S3, with utility functions to flatten and process nested structures.
   - S3 keys are organized by user, area, route, and timestamp, allowing efficient search and listing via API endpoints.

2. **Temporary & Output Files**
   - Uploaded images, intermediate JSON files, and output videos are stored in `temp_uploads/` during processing.
   - Temporary files are cleaned up via dedicated API endpoints to prevent storage bloat and ensure privacy.
   - Output files (videos, pose data) are moved to `static/` for serving or uploaded to S3 for long-term storage.

3. **Database Connectivity**
   - MySQL is used for persistent storage of user and route metadata, with connections managed via environment variables for security.
   - All credentials are loaded from `.env` files or secret managers, never hard-coded.

4. **Amazon S3 Integration**
   - S3 is used for storing pose landmarks, SIFT keypoints, cached location trees, and user route data.
   - Data is organized by user prefix, with utility functions to build, cache, and convert S3 key trees for fast access.
   - S3 bucket policies enforce private access; presigned URLs or API proxies are used for secure delivery.

5. **Data Validation & Cleanup**
   - All loaded data (JSON, images, coordinates) is validated for integrity and format before use.
   - Cleanup routines ensure that stale or unused files are removed from both local and S3 storage.

This approach ensures fast, secure, and scalable handling of climbing route data, pose features, and video outputs, supporting both interactive browsing and automated processing workflows.

---

## Deployment

### Dockerized Backend

The backend runs in a Docker container for reproducibility.

**Build:**

```bash
docker build -t route-backend .
```

**Run locally:**

```bash
docker run -p 8000:8000 --env-file .env route-backend
```

### Production Hosting

Deploy with AWS ECS, EKS, or EC2.
Serve via uvicorn behind a production server such as NGINX or Traefik for reverse proxy and SSL termination.

### Static Files & Media

Reference frames and output videos are stored temporarily in `temp_uploads/`.
Processed JSON files and final outputs are uploaded to Amazon S3.
An S3 bucket policy enforces private access; presigned URLs or API proxies are used to deliver files securely.

### CI/CD

Automated tests and linting run via GitHub Actions before deploys.
Docker images are built and pushed to AWS ECR.
Deployment is triggered to ECS/EKS using IaC (Terraform or CloudFormation recommended).

### Scaling

Workers can be scaled horizontally to handle multiple video processing jobs.
Long-running tasks run asynchronously in background jobs (tracked by `job_manager`).

---

## Security

### Environment Variables

Secrets (AWS keys, DB credentials, API tokens) must never be hard-coded.

Store them in .env files (excluded from version control) or your deployment platform’s secret manager (e.g., AWS Secrets Manager, Docker secrets).

### Access Control

Ensure all API routes that expose user or job data require authentication.

Use short-lived tokens or session-based authentication rather than long-lived static keys.

### File Handling

Uploaded files are stored temporarily in temp_uploads/ and cleaned up after processing.

Always validate file size and type before processing to prevent malicious uploads.

### Network & Data

Run behind HTTPS (TLS) in production.

Use AWS IAM roles for S3 access instead of static credentials where possible.

Restrict CORS to trusted front-end domains.

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
```
