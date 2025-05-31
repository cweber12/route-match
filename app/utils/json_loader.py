import json

def load_json_from_path(path):
    with open(path, "r") as f:
        return json.load(f)