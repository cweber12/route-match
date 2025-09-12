# BACKEND MATCHING AND VIDEO OUTPUT

_______________________________________________________

## RouteMap Backend Match

## Directory Contents

### app/

- main.py

#### api/

##### routers/
  
###### auth.py

- @router.post("/register")
- @router.post("/login")

###### browse_user_routes.py

- @router.get("/s3-location-tree")
- @router.get("/recent-attempts")
- @router.get("/all-route-coordinates")
- @router.get("/s3-search")
- @router.get("/list-coordinates")
- @router.get("/list-timestamps")
- @router.get("/debug-list-s3-keys")
- @router.get("/routes-under-area")

###### compare.py

- @router.post("/compare-image")
- @router.get("/compare-status/{job_id}")

###### health.py

- @router.get("/health")
- @router.get("/health-check-fs")

###### map_data.py

- @router.get("/map-data")
  
###### stream_frames.py

- @router.post("/stream-frames")

###### temp_cleanup.py

- @router.delete("/clear-temp")
- @router.post("/clear-output")

#### jobs/

- job_manager.py

#### storage/

##### database/

- route_db_connect.py

##### local/

- json_loader.py
- temp_file_storage.py

##### s3/

- cache_s3_loc_tree.py
- load_json_s3.py
- tree_helpers.py

#### transform/

- draw_points.py
- transform_skeleton.py

#### video/

- video_writer.py

#### vision/

- detect_img_sift.py
- match_features.py

## CLI Commands

### DOCKER (LOCAL)

#### STOP AND REMOVE

```bash
docker stop route-map-match

docker rm route-map-match
```

#### BUILD AND RUN

```bash
docker build -t route-map-match .

docker run --env-file .env -p 8000:8000 --name route-map-match route-map-match
```

docker ps

docker system prune

### AWS ELASTIC CONTAINER REGISTRY (ECR)

#### LOGIN

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

### AWS ELASTIC COMPUTE (EC2) 

#### OPEN SUCURE SHELL TO EC2 INSTANCE

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
