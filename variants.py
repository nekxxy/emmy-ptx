"""Typed IR mutations: different pointwise shapes/ops and reduction lengths.
Run against a separate official task checkout. No reference compiler is called.
"""

import argparse
import dataclasses as dc
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import requests
from emmy.compiler.graph import Graph
from emmy.compiler.dim import Dim
from emmy.compiler.ir.elementwise import ElementwiseImpl

ap = argparse.ArgumentParser()
ap.add_argument("tasks", type=Path)
ap.add_argument("--server", default="http://runner.riftstack.ai:48701")
opts = ap.parse_args()
compiler = Path(__file__).resolve().parent / "compile.py"
rng = np.random.default_rng(401)


def mutate(body, *, size=None, op=None, tile=None):
    out = []
    for s in body:
        if type(s).__name__ == "Tile" and tile is not None:
            s = dc.replace(s, axes=(dc.replace(s.axes[0], extent=Dim(tile)),))
        if type(s).__name__ == "StridedLoop" and size is not None:
            s = dc.replace(s, axis=dc.replace(s.axis, extent=Dim(size)))
        if type(s).__name__ == "Assign" and op is not None:
            impl = ElementwiseImpl(op)
            s = dc.replace(s, op=impl, args=s.args[: impl.arity])
        for key in ["body", "else_body"]:
            if hasattr(s, key):
                s = dc.replace(
                    s, **{key: mutate(getattr(s, key), size=size, op=op, tile=None)}
                )
        out.append(s)
    return tuple(out)


failed = 0
for kind in ["pointwise", "reduce"]:
    source = opts.tasks / ("p00_scalar_add" if kind == "pointwise" else "r01_array_sum")
    for n in (
        [1, 17, 65, 129, 513] if kind == "pointwise" else [1, 31, 129, 1003, 8191]
    ):
        for op in (
            [
                "add",
                "subtract",
                "multiply",
                "maximum",
                "minimum",
                "negative",
                "abs",
                "tanh",
                "sigmoid",
                "silu",
            ]
            if kind == "pointwise"
            else ["add"]
        ):
            g = Graph.from_dict(json.loads(next(source.glob("*.ir.json")).read_text()))
            meta = json.loads((source / "meta.json").read_text())
            k = meta["kernels"][0]
            entry = f"variant_{kind}_{n}_{op}"
            k["entry"] = entry
            outname = meta["graph_outputs"][0]
            size = n * 2 if kind == "pointwise" else n
            for node in g.nodes.values():
                node.outputs = tuple(
                    (
                        dc.replace(t, shape=(Dim(size),))
                        if t.name != outname or kind == "pointwise"
                        else t
                    )
                    for t in node.outputs
                )
                if type(node.op).__name__ == "KernelOp":
                    node.op = dc.replace(
                        node.op,
                        name=entry,
                        body=mutate(
                            node.op.body,
                            size=n if kind == "reduce" else None,
                            tile=n if kind == "pointwise" else None,
                            op=op if kind == "pointwise" else None,
                        ),
                    )
            k["grid"] = (
                [math.ceil(n / k["block"][0]), 1, 1]
                if kind == "pointwise"
                else k["grid"]
            )
            args, ins = [], []
            for a in k["args"]:
                a = dict(a)
                if not a.get("output") or kind == "pointwise":
                    a["shape"] = [size]
                if not a.get("output"):
                    x = rng.standard_normal(a["shape"]).astype(np.float32)
                    ins.append(x)
                    a.pop("seed", None)
                    a["data"] = x.tolist()
                args.append(a)
            if kind == "pointwise":
                impl = ElementwiseImpl(op)
                ref = impl(*ins[: impl.arity])
            else:
                ref = ins[0].sum(keepdims=True)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "kernel.ir.json"
                p.write_text(json.dumps(g.to_dict(), default=repr))
                (p.parent / "meta.json").write_text(json.dumps(meta))
                ptx = subprocess.check_output(
                    [sys.executable, str(compiler), str(p)], text=True
                )
            r = requests.post(
                opts.server + "/run",
                json={
                    "ptx": ptx,
                    "entry": entry,
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
                print(entry, "ERROR", r)
                failed += 1
                continue
            got = np.array(r["outputs"][outname]["values"], dtype=np.float32).reshape(
                ref.shape
            )
            rtol, atol = meta["rtol"], meta["atol"]
            if op in ["tanh", "sigmoid", "silu"]:
                # These invented op variants have no official tolerance. Report
                # an explicit approximate-op test budget, never call it official.
                rtol, atol = 1e-4, 1e-5
            ok = np.allclose(got, ref, rtol=rtol, atol=atol)
            print(
                entry,
                "PASS" if ok else "FAIL",
                "max_error",
                float(np.max(np.abs(got - ref))),
                "rtol",
                rtol,
                "atol",
                atol,
            )
            failed += not ok
sys.exit(bool(failed))
