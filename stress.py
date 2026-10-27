"""Fresh-input GPU tests. Reads an independently obtained official task checkout.
No official problem data is redistributed. Test references are NumPy, not the compiler.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument("tasks", type=Path)
ap.add_argument("--server", default="http://runner.riftstack.ai:48701")
opts = ap.parse_args()
rng = np.random.default_rng(20261027)
compiler = Path(__file__).resolve().parent / "compile.py"
failed = 0
for folder in sorted(opts.tasks.iterdir()):
    if not (folder / "meta.json").exists():
        continue
    meta = json.loads((folder / "meta.json").read_text())
    # The open problems currently contain one kernel each. Reject, don't fake chaining.
    assert len(meta["kernels"]) == 1
    k = meta["kernels"][0]
    args, arrays = [], {}
    for a in k["args"]:
        a = dict(a)
        if not a.get("output") and "seed" in a:
            x = rng.standard_normal(a["shape"]).astype(np.float32) * 0.3
            a.pop("seed")
            a["data"] = x.reshape(-1).tolist()
        elif "data" in a:
            x = np.array(a["data"], dtype=np.float32).reshape(a["shape"])
        else:
            x = np.zeros(a["shape"], dtype=np.float32)
        arrays[a["name"]] = x
        args.append(a)
    name = folder.name
    x = arrays.get("x")
    if name in ["p00_scalar_add", "p01_vector_add"]:
        ins = [arrays[a] for a in meta["graph_inputs"]]
        ref = ins[0] + ins[1]
    elif name == "p02_scale_shift":
        ref = x * arrays["mul_c1"] + arrays["add_c1"]
    elif name == "p03_relu_mul":
        ins = [arrays[a] for a in meta["graph_inputs"]]
        ref = np.maximum(ins[0], 0) * ins[1]
    elif name in ["r01_array_sum", "r02_big_sum"]:
        ref = np.sum(x, keepdims=True)
    elif name == "r03_mean":
        ref = np.mean(x, keepdims=True)
    elif name == "r04_row_sum":
        ref = np.sum(x, axis=-1)
    elif name == "f01_rmsnorm":
        ref = (
            x
            / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + arrays["add_c1"])
            * arrays["w"]
        )
    elif name == "f02_row_softmax":
        exp = np.exp(x - x.max(axis=-1, keepdims=True))
        ref = exp / exp.sum(axis=-1, keepdims=True)
    elif name == "f03_l2norm":
        ref = x / np.sqrt((x * x).sum(axis=-1, keepdims=True))
    elif name.startswith("m0"):
        ref = arrays["x0"] @ arrays["x1"]
    elif name.startswith("a0"):
        z = (arrays["q"] @ arrays["kt"]) / arrays["div_c1"]
        z = np.exp(z - z.max(axis=-1, keepdims=True))
        ref = (z / z.sum(axis=-1, keepdims=True)) @ arrays["v"]
    else:
        raise ValueError(name)
    ptx = subprocess.check_output(
        [sys.executable, str(compiler), str(next(folder.glob("*.ir.json")))], text=True
    )
    r = requests.post(
        opts.server + "/run",
        json={
            "ptx": ptx,
            "entry": k["entry"],
            "grid": k["grid"],
            "block": k["block"],
            "smem_bytes": k["smem_bytes"],
            "zero_outputs": k["zero_outputs"],
            "args": args,
            "iters": 1,
            "warmup": 0,
            "max_return": max(4096, ref.size),
        },
        timeout=90,
    ).json()
    if not r.get("ok"):
        print(name, "ERROR", r)
        failed += 1
        continue
    out = r["outputs"][meta["graph_outputs"][0]]
    got = np.array(out["values"], dtype=np.float32).reshape(ref.shape)
    ok = np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"])
    print(name, "PASS" if ok else "FAIL", "max_error", float(np.max(np.abs(got - ref))))
    failed += not ok
sys.exit(bool(failed))
