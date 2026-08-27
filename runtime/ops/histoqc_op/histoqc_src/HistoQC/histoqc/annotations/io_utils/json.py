import json


def write_json(fname: str, obj, **kwargs):
    with open(fname, 'w') as root:
        json.dump(obj, root, **kwargs)


def load_json(fname: str, **kwargs):
    with open(fname, 'r') as root:
        return json.load(root, **kwargs)

