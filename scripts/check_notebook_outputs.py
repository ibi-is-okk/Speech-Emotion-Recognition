import json, glob
for path in sorted(glob.glob("notebooks/*.ipynb")):
    nb = json.load(open(path, encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    with_out = [c for c in code if c.get("outputs")]
    print(f"{path}: {len(with_out)} of {len(code)} code cells have outputs")