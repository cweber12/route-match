# app/routers/browse_user_routes.py
# ------------------------------------------------------------------------
# This file handles browsing user routes, including: 
#   - Fetching the cached route/area tree from the user's s3 folder to reduce
#     route/area loading times. 
#   - Fetching recent attempts and route coordinates from S3.
# ------------------------------------------------------------------------

from fastapi import APIRouter, Query, HTTPException
import boto3
import os
import json
from datetime import datetime
import re

from app.utils.cache_s3_loc_tree import cache_full_location_tree
from app.utils.tree_helpers import build_tree_from_s3_keys, dict_to_node_array, add_path_to_tree

router = APIRouter()

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# -----------------------------------------------------------------------
# Utility functions (utils/cache_s3_loc_tree.py, utils/tree_helpers.py)
# -----------------------------------------------------------------------

# Try to load coordinates from S3
def try_load_coordinates(bucket, prefix):
    key = prefix.rstrip("/") + "/gps_location/coordinates.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return {}

# Flatten coordinates from a list of nodes
# This function is used to convert a nested structure of nodes into a flat list
def flatten_coords(nodes, prefix=""):
    print(f"DEBUG: flatten_coords called with prefix={prefix}, nodes={len(nodes)}")
    flat = []
    for node in nodes:
        if not isinstance(node, dict):
            print(f"DEBUG: flatten_coords skipping non-dict node: {node}")
            continue  # Skip if node is not a dict
        if node.get("latitude") is not None and node.get("longitude") is not None:
            flat.append({
                "name": node["name"],
                "path": node["path"],
                "type": node["type"],
                "label": prefix + " / " + node["name"] if prefix else node["name"],
                "lat": node["latitude"],
                "lng": node["longitude"]
            })
            print(f"DEBUG: flatten_coords added node: {node['name']}")
        if node.get("children"):
            print(f"DEBUG: flatten_coords recursing into children of {node['name']}")
            flat.extend(flatten_coords(node["children"], prefix + " / " + node["name"]))
    print(f"DEBUG: flatten_coords returning {len(flat)} flat nodes for prefix={prefix}")
    return flat

# Load a previously cached location tree from S3 or build it if not present
def load_or_build_tree(user, s3_client, bucket="route-keypoints"):
    import json
    from botocore.exceptions import ClientError

    tree_key = f"{user}/location_tree.json"
    print(f"DEBUG: load_or_build_tree for user={user}, tree_key={tree_key}")
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=tree_key)
        print(f"DEBUG: load_or_build_tree loaded cached tree for user={user}")
        tree = json.loads(obj["Body"].read())
        return tree if isinstance(tree, list) else []
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"DEBUG: load_or_build_tree cache miss for user={user}, building tree")
            return cache_full_location_tree(user, s3_client, bucket)
        else:
            print(f"DEBUG: load_or_build_tree error for user={user}: {e}")
            raise

# This function adds a new path to the existing tree structure.
def add_path_to_tree(tree, path_parts):
    node = tree
    for part in path_parts:
        part = part.strip()
        if not part:
            continue
        if part not in node:
            node[part] = {}
        node = node[part]

# Build a tree from a list of S3 keys
def build_tree_from_s3_keys(keys, prefix_len):
    tree = {}
    for key in keys:
        parts = key.strip("/").split("/")[prefix_len:]
        add_path_to_tree(tree, parts)
    return tree

TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

# Check if a folder name matches the timestamp pattern
# Timestamp folders represent stored route attempts inside route folders
def is_timestamp_folder(name):
    return bool(TIMESTAMP_PATTERN.fullmatch(name))

# Convert a dictionary tree structure into an array of nodes
# This function is used to convert a nested structure of nodes into an array format for processing
def dict_to_node_array(tree, prefix="", bucket="route-keypoints", try_load_coordinates=None, s3_client=None):
    nodes = []
    for k, v in tree.items():
        node_prefix = f"{prefix}/{k}" if prefix else k
        coords = {}
        if try_load_coordinates:
            coords = try_load_coordinates(bucket, node_prefix)

        # Filter out timestamp folders and files
        child_keys = list(v.keys())
        timestamp_children = [name for name in child_keys if is_timestamp_folder(name)]
        file_children = [name for name in child_keys if '.' in name]  # crude file check
        non_timestamp_non_file_children = [
            name for name in child_keys
            if not is_timestamp_folder(name) and '.' not in name
        ]

        # If this folder has children, BUT no sub-folders (areas), AND it has timestamp folders or files, then it's a ROUTE
        if child_keys and len(non_timestamp_non_file_children) == 0 and (len(timestamp_children) > 0 or len(file_children) > 0):
            node = {
                "name": k,
                "type": "route",
                "latitude": coords.get("lat"),
                "longitude": coords.get("lng"),
                "children": []
            }
        # Otherwise, it's an AREA (contains sub-areas or sub-routes)
        else:
 
            children = dict_to_node_array(
                {name: v[name] for name in non_timestamp_non_file_children},
                node_prefix,
                bucket,
                try_load_coordinates,
                s3_client
            )
            node = {
                "name": k,
                "type": "area",
                "latitude": coords.get("lat"),
                "longitude": coords.get("lng"),
                "children": children
            }
        nodes.append(node)
    return nodes

# -----------------------------
# Routes
# -----------------------------

# Fetch the S3 location tree for a user
@router.get("/s3-location-tree")
def get_s3_location_tree(user: str = Query(...), bucket: str = Query("route-keypoints")):
    # Always returns an array
    return load_or_build_tree(user, s3_client, bucket)

# Fetch recent route attempts for a user
@router.get("/recent-attempts")
def get_recent_attempts(user: str = Query(...), bucket: str = Query("route-keypoints")):
    prefix = f"{user}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    attempts = []

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "pose" in key and key.endswith(".json"):
                parts = key.split("/")
                if len(parts) >= 4:
                    try:
                        timestamp = parts[-2]
                        parsed_time = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
                        route_name = parts[-3]
                        base_path = "/".join(parts[:-2])  # up to route
                        attempts.append({
                            "timestamp": timestamp,
                            "parsed_time": parsed_time,
                            "route_name": route_name,
                            "basePath": base_path
                        })
                    except ValueError:
                        continue

    sorted_attempts = sorted(attempts, key=lambda x: x["parsed_time"], reverse=True)
    return sorted_attempts[:10]

# Fetch all route coordinates for a user
@router.get("/all-route-coordinates")
def get_all_route_coordinates(
    user: str = Query(...), 
    bucket: str = Query("route-keypoints")
):
    try:
        print(f"DEBUG: all-route-coordinates processing user: {user}")
        user_tree = load_or_build_tree(user, s3_client, bucket)
        flat = flatten_coords(user_tree)
        print(f"DEBUG: all-route-coordinates user {user} added {len(flat)} locations")
        return flat
    except Exception as e:
        print(f"Error building tree for {user}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch route coordinates: {e}")

# -----------------------------
# S3 Search and Listing 
# -----------------------------

# Search S3 for keys matching a prefix
@router.get("/s3-search")
def s3_search(
    bucket: str = Query("route-keypoints"),
    prefix: str = Query("", description="S3 key prefix to search under"),
    delimiter: str = Query("/", description="Delimiter to group 'folders'"),
    max_keys: int = Query(10, description="Max suggestions to return")
):
    resp = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter=delimiter,
        MaxKeys=max_keys
    )
    suggestions = []
    for cp in resp.get("CommonPrefixes", []):
        p = cp["Prefix"].rstrip("/")   # e.g. "user/Area/Sub/"
        name = p.split("/")[-1]
        suggestions.append({
            "name": name,
            "path": p.split("/"),       # will be an array
        })
    return suggestions

# List all coordinates for a given prefix
@router.get("/list-coordinates")
def list_coordinates(
    prefix: str = Query(..., description="S3 key prefix, e.g. 'user/Area/Subarea'"),
    bucket: str = Query("route-keypoints", description="S3 bucket name")
):
    key = prefix.rstrip("/") + "/gps_location/coordinates.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        raise HTTPException(status_code=404, detail="Coordinates not found")

# List all timestamp folders under a given prefix
@router.get("/list-timestamps")
def list_timestamps(
    prefix: str = Query(..., description="S3 key prefix, e.g. 'user/Area/Subarea/Route'"),
    bucket: str = Query("route-keypoints", description="S3 bucket name")
):
    paginator = s3_client.get_paginator("list_objects_v2")
    # No Delimiter!
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/")
    timestamp_folders = set()
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Remove the prefix and split by "/"
            rel = key[len(prefix.rstrip("/") + "/"):]
            parts = rel.split("/")
            if len(parts) > 1:
                folder = parts[0]
                if is_timestamp_folder(folder):
                    timestamp_folders.add(folder)
    timestamps = [{"name": t} for t in sorted(timestamp_folders)]
    print("DEBUG: Returning timestamps:", timestamps)
    return timestamps

# Debugging route to list S3 keys
@router.get("/debug-list-s3-keys")
def debug_list_s3_keys(
    prefix: str = Query(...),
    bucket: str = Query("route-keypoints")
):
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/")
    keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    print("DEBUG: S3 KEYS:", keys)
    return keys

# Fetch routes under a specific area
# This route returns all routes (timestamp folders) under a given area path
@router.get("/routes-under-area")
def routes_under_area(user: str = Query(...), area_path: str = Query(...), bucket: str = Query("route-keypoints")):
    # Load the user's location tree from S3
    tree = load_or_build_tree(user, s3_client, bucket)
    path_parts = [p for p in area_path.strip("/").split("/") if p]

    # Traverse to the area node
    def find_area_node(nodes, parts):
        if not parts:
            return nodes
        for node in nodes:
            if node["name"] == parts[0] and node["type"] == "area":
                return find_area_node(node.get("children", []), parts[1:])
        return []

    area_nodes = find_area_node(tree, path_parts)

    # Collect all routes (leaf nodes) under this area
    def collect_routes(nodes):
        routes = []
        for node in nodes:
            if node["type"] == "route":
                # Fetch timestamps for this route
                prefix = f"{user}/{area_path}/{node['name']}".replace('//', '/')
                timestamps = []
                try:
                    paginator = s3_client.get_paginator("list_objects_v2")
                    pages = paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/")
                    timestamp_folders = set()
                    for page in pages:
                        for obj in page.get("Contents", []):
                            key = obj["Key"]
                            rel = key[len(prefix.rstrip('/') + '/'):]
                            parts = rel.split('/')
                            if len(parts) > 1:
                                folder = parts[0]
                                if is_timestamp_folder(folder):
                                    timestamp_folders.add(folder)
                    timestamps = [{"name": t} for t in sorted(timestamp_folders)]
                except Exception:
                    timestamps = []
                node_copy = node.copy()
                node_copy["timestamps"] = timestamps
                routes.append(node_copy)
            elif node.get("children"):
                routes.extend(collect_routes(node["children"]))
        return routes

    return collect_routes(area_nodes)