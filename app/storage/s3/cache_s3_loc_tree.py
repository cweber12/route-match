import json
import os
from app.storage.s3.tree_helpers import build_tree_from_s3_keys, dict_to_node_array

def cache_full_location_tree(user, s3_client, bucket="route-keypoints"):
    """
    Build the full location tree for a user and cache it as location_tree.json in S3.
    Returns the tree as an array of nodes.

    Args:
        user (str): The user name or prefix.
        s3_client: A boto3 S3 client.
        bucket (str): S3 bucket name.
    """
    import tempfile

    # List all keys for this user
    prefix = f"{user}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    # Build the nested dict tree
    tree_dict = build_tree_from_s3_keys(keys, prefix_len=1)  # skip the user part
    # Convert to array-of-nodes for frontend
    tree_array = dict_to_node_array(tree_dict)
    # Save to S3
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        json.dump(tree_array, tmp)
        tmp_path = tmp.name
    tree_key = f"{user}/location_tree.json"
    s3_client.upload_file(tmp_path, bucket, tree_key)
    os.remove(tmp_path)
    return tree_array