import json, pickle, pathlib, datetime as dt

def timestamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def dump_json(obj, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def dump_pickle(obj, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
