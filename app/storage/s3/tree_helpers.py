import re

TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

def is_timestamp_folder(name):
    return bool(TIMESTAMP_PATTERN.fullmatch(name))

def add_path_to_tree(tree, path_parts):
    node = tree
    for part in path_parts:
        part = part.strip()
        if not part:
            continue
        if part not in node:
            node[part] = {}
        node = node[part]

def build_tree_from_s3_keys(keys, prefix_len):
    tree = {}
    for key in keys:
        parts = key.strip("/").split("/")[prefix_len:]
        add_path_to_tree(tree, parts)
    return tree

def dict_to_node_array(tree, prefix="", bucket="route-keypoints", try_load_coordinates=None, s3_client=None):
    nodes = []
    for k, v in tree.items():
        node_prefix = f"{prefix}/{k}" if prefix else k
        coords = {}
        if try_load_coordinates:
            coords = try_load_coordinates(bucket, node_prefix)
        # If this node is a timestamp folder, mark as type "timestamp" and do not recurse
        if is_timestamp_folder(k):
            node = {
                "name": k,
                "type": "timestamp",
                "latitude": coords.get("lat"),
                "longitude": coords.get("lng"),
                "children": []
            }
            nodes.append(node)
            continue
        # Otherwise, recurse for children
        children = dict_to_node_array(v, node_prefix, bucket, try_load_coordinates, s3_client)
        node_type = "area"
        if children and all(child["type"] == "timestamp" for child in children):
            node_type = "route"
        node = {
            "name": k,
            "type": node_type,
            "latitude": coords.get("lat"),
            "longitude": coords.get("lng"),
            "children": children
        }
        nodes.append(node)
    return nodes