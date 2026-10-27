"""Original typed Kernel IR to PTX lowering. No existing renderer/compiler used."""

import json
import sys
import struct
import math
from pathlib import Path
from emmy.compiler.graph import Graph


def dim(x):
    return int(x.expr.value) if hasattr(x, "expr") else int(x)


class Compiler:
    def __init__(self, g, k, meta):
        self.g, self.k, self.meta = g, k, meta
        self.lines = []
        self.count = {t: 0 for t in ["r", "rd", "f", "p"]}
        self.env = {}
        self.ptr = {}
        self.shared = {}
        self.shared_offsets = {}
        self.shared_alignments = {}
        self.labels = 0

    def reg(self, t="r"):
        self.count[t] += 1
        return "%" + t + str(self.count[t])

    def emit(self, s):
        self.lines.append("    " + s)

    def label(self):
        self.labels += 1
        return "L" + str(self.labels)

    def set(self, name, reg):
        self.env[name] = reg
        return reg

    def mov(self, x, t="r"):
        r = self.reg(t)
        self.emit(f'mov.{dict(r="u32",rd="u64",f="f32",p="pred")[t]} {r}, {x};')
        return r

    def lit(self, v, t="r"):
        return (
            "0f" + struct.pack(">f", float(v)).hex().upper()
            if t == "f"
            else str(int(v))
        )

    def expr(self, e, t="r"):
        if isinstance(e, str):
            return self.env[e]
        if isinstance(e, (int, float)):
            return self.lit(e, t)
        n = type(e).__name__
        if n == "Var":
            return self.env[e.name]
        if n == "Literal":
            return self.lit(e.value, t)
        if n == "BinaryExpr":
            a, b = self.expr(e.left, t), self.expr(e.right, t)
            ops = {
                "+": "add",
                "-": "sub",
                "*": "mul.lo",
                "//": "div",
                "%": "rem",
                "/": "div",
            }
            cmp = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
            if e.op in cmp:
                r = self.reg("p")
                self.emit(f"setp.{cmp[e.op]}.s32 {r}, {a}, {b};")
                return r
            if e.op in ["and", "&&", "or", "||"]:
                r = self.reg("p")
                self.emit(
                    f'{"and" if e.op in ["and","&&"] else "or"}.pred {r}, {a}, {b};'
                )
                return r
            r = self.reg(t)
            self.emit(f"{ops[e.op]}.s32 {r}, {a}, {b};")
            return r
        if n == "TernaryExpr":
            p = self.expr(e.cond)
            a, b = self.expr(e.if_true, t), self.expr(e.if_false, t)
            r = self.reg(t)
            self.emit(f'selp.{"f32" if t=="f" else "u32"} {r}, {a}, {b}, {p};')
            return r
        raise NotImplementedError(("expression", repr(e)))

    def offset(self, buf, indices):
        shape = (
            self.shared[buf]
            if buf in self.shared
            else next(
                t.shape
                for n in self.g.nodes.values()
                for t in n.outputs
                if t.name == buf
            )
        )
        if not indices:
            return "0"
        vals = [self.expr(i) for i in indices]
        o = vals[0]
        for i in range(1, len(vals)):
            r = self.reg()
            self.emit(f"mad.lo.s32 {r}, {o}, {dim(shape[i])}, {vals[i]};")
            o = r
        return o

    def addr(self, buf, indices, extra=0):
        o = self.offset(buf, indices)
        if extra:
            r = self.reg()
            self.emit(f"add.u32 {r}, {o}, {extra};")
            o = r
        if buf in self.shared:
            b = self.reg()
            self.emit(f"mul.lo.u32 {b}, {o}, 4;")
            a = self.reg()
            base = self.reg()
            self.emit(
                f"mov.u32 {base}, "
                + ("_dynamic_smem" if self.meta.get("smem_bytes", 0) > 49152 else buf)
                + ";"
            )
            if self.meta.get("smem_bytes", 0) > 49152:
                self.emit(f"add.u32 {base}, {base}, {self.shared_offsets[buf]};")
            self.emit(f"add.u32 {a}, {base}, {b};")
            return a, "shared"
        b = self.reg("rd")
        self.emit(f"mul.wide.u32 {b}, {o}, 4;")
        a = self.reg("rd")
        self.emit(f"add.u64 {a}, {self.ptr[buf]}, {b};")
        return a, "global"

    def op(self, op, args, dest=None):
        op = getattr(op, "name", str(op))
        op = {
            "sum": "add",
            "prod": "multiply",
            "amax": "maximum",
            "exp_fast": "exp",
            "pad": "copy",
        }.get(op, op)
        args = [
            self.env.get(a, a) if isinstance(a, str) else self.expr(a, "f")
            for a in args
        ]
        r = dest or self.reg("f")
        ops = {
            "add": "add.f32",
            "subtract": "sub.f32",
            "multiply": "mul.f32",
            "divide": "div.rn.f32",
            "maximum": "max.f32",
            "minimum": "min.f32",
            "neg": "neg.f32",
            "negative": "neg.f32",
            "sqrt": "sqrt.rn.f32",
            "rsqrt": "rsqrt.approx.f32",
            "abs": "abs.f32",
            "reciprocal": "rcp.rn.f32",
            "copy": "mov.f32",
        }
        if op == "relu":
            self.emit(f"max.f32 {r}, {args[0]}, 0f00000000;")
        elif op in ["exp", "exp2"]:
            x = args[0]
            if op == "exp":
                x = self.reg("f")
                self.emit(f'mul.f32 {x}, {args[0]}, {self.lit(math.log2(math.e),"f")};')
            self.emit(f"ex2.approx.f32 {r}, {x};")
        elif op in ["log", "log2"]:
            tmp = self.reg("f") if op == "log" else r
            self.emit(f"lg2.approx.f32 {tmp}, {args[0]};")
            if op == "log":
                self.emit(f'mul.f32 {r}, {tmp}, {self.lit(math.log(2), "f")};')
        elif op == "tanh":
            self.emit(f"tanh.approx.f32 {r}, {args[0]};")
        elif op in ["sigmoid", "silu"]:
            neg, exp, den = self.reg("f"), self.reg("f"), self.reg("f")
            self.emit(f"neg.f32 {neg}, {args[0]};")
            self.op("exp", [neg], exp)
            self.emit(f'add.f32 {den}, {exp}, {self.lit(1, "f")};')
            self.emit(
                f'div.rn.f32 {r}, {args[0] if op == "silu" else self.lit(1, "f")}, {den};'
            )
        elif op in ops:
            self.emit(f"{ops[op]} {r}, " + ", ".join(args) + ";")
        else:
            raise NotImplementedError(("operation", op, args))
        return r

    def walk(self, body):
        for s in body:
            self.stmt(s)

    def scan_accum(self, body):
        for s in body:
            if type(s).__name__ == "Accum" and s.name not in self.env:
                op = getattr(s.op, "name", str(s.op))
                v = s.op.identity
                if v is None:
                    raise NotImplementedError(("reduction identity", op))
                self.set(s.name, self.mov(self.lit(v, "f"), "f"))
            for attr in ["body", "else_body"]:
                if hasattr(s, attr):
                    self.scan_accum(getattr(s, attr))

    def reset_loop(self, loop):
        if not loop.seed:
            return

        def visit(body):
            for a in body:
                if (
                    type(a).__name__ == "Accum"
                    and loop.axis.name in a.axes
                    and a.name not in self.explicit_inits
                ):
                    op = a.op.name
                    v = a.op.identity
                    if v is None:
                        raise NotImplementedError(("reduction identity", op))
                    self.emit(f'mov.f32 {self.env[a.name]}, {self.lit(v,"f")};')
                for key in ["body", "else_body"]:
                    if hasattr(a, key):
                        visit(getattr(a, key))

        visit(loop.body)

    def stmt(self, s):
        n = type(s).__name__
        if n == "Tile":
            if s.aux_threads:
                raise NotImplementedError(
                    "auxiliary producer threads are outside this backend"
                )
            tid = self.mov("%tid.x")
            bid = self.mov("%ctaid.x")
            nt = self.mov("%ntid.x")
            linear = self.reg()
            self.emit(f"mad.lo.u32 {linear}, {bid}, {nt}, {tid};")
            total = math.prod(dim(a.extent) for a in s.axes)
            p = self.reg("p")
            self.emit(f"setp.ge.u32 {p}, {linear}, {total};")
            self.emit(f"@{p} ret;")
            rem = linear
            for a in reversed(s.axes):
                r = self.reg()
                self.emit(f"rem.u32 {r}, {rem}, {dim(a.extent)};")
                self.set(a.name, r)
                d = self.reg()
                self.emit(f"div.u32 {d}, {rem}, {dim(a.extent)};")
                rem = d
            self.set("lane", self.reg())
            self.emit(f'and.b32 {self.env["lane"]}, {tid}, 31;')
            self.set("warp", self.reg())
            self.emit(f'shr.u32 {self.env["warp"]}, {tid}, 5;')
            self.walk(s.body)
        elif n == "Load":
            for i, name in enumerate(s.names):
                a, space = self.addr(s.input, s.index, i)
                r = self.reg("f")
                self.emit(f"ld.{space}.f32 {r}, [{a}];")
                self.set(name, r)
        elif n == "Write":
            for i, v in enumerate(s.values):
                a, space = self.addr(s.output, s.index, i)
                v = self.env[v]
                self.emit(
                    f'{"red.global.add.f32" if s.atomic else "st."+space+".f32"} [{a}], {v};'
                )
        elif n == "Assign":
            self.set(
                s.name,
                self.op(
                    s.op,
                    s.args,
                    self.env.get(s.name) if s.name in self.accums else None,
                ),
            )
        elif n == "Let":
            self.set(
                s.name,
                self.mov(
                    self.expr(
                        s.value,
                        "f" if getattr(s.dtype, "name", s.dtype) == "f32" else "r",
                    ),
                    "f" if getattr(s.dtype, "name", s.dtype) == "f32" else "r",
                ),
            )
        elif n == "Init":
            self.set(s.name, self.mov(self.lit(s.identity, "f"), "f"))
        elif n == "Accum":
            self.op(s.op, [s.base or s.name, s.value], self.env[s.name])
        elif n in ["StridedLoop", "Loop"]:
            self.scan_accum(s.body)
            self.reset_loop(s)
            r = self.reg()
            start = self.expr(s.start) if n == "StridedLoop" else "0"
            step = self.expr(s.step) if n == "StridedLoop" else "1"
            end = (
                self.expr(s.end)
                if n == "StridedLoop" and s.end is not None
                else str(dim(s.axis.extent))
            )
            self.emit(f"mov.u32 {r}, {start};")
            self.set(s.axis.name, r)
            top, done = self.label(), self.label()
            self.emit(top + ":")
            p = self.reg("p")
            self.emit(f"setp.ge.s32 {p}, {r}, {end};")
            self.emit(f"@{p} bra {done};")
            self.walk(s.body)
            self.emit(f"add.s32 {r}, {r}, {step};")
            self.emit(f"bra {top};")
            self.emit(done + ":")
        elif n == "Cond":
            p = self.expr(s.cond)
            no, end = self.label(), self.label()
            self.emit(f"@!{p} bra {no};")
            self.walk(s.body)
            self.emit(f"bra {end};")
            self.emit(no + ":")
            self.walk(s.else_body)
            self.emit(end + ":")
        elif n == "WarpShuffle":
            d = s.length // 2
            while d:
                for a, b in zip(s.state, s.state_b):
                    tmp = self.reg("f")
                    self.emit(
                        f"shfl.sync.bfly.b32 {tmp}, {self.env[a]}, {d}, 31, 0xffffffff;"
                    )
                    self.set(b, tmp)
                self.walk(s.combine_states)
                d //= 2
        elif n == "Smem":
            alignment = max(4, s.align or 4)
            end = max(
                (
                    self.shared_offsets[name] + math.prod(shape) * 4
                    for name, shape in self.shared.items()
                ),
                default=0,
            )
            self.shared_offsets[s.name] = (end + alignment - 1) // alignment * alignment
            self.shared_alignments[s.name] = alignment
            self.shared[s.name] = s.extents
        elif n == "CpAsyncCopy":
            if str(s.swizzle) != "NONE":
                raise NotImplementedError(("shared swizzle", s.swizzle))
            from emmy.compiler.ir.expr import BinaryExpr

            indices = (
                tuple(BinaryExpr("+", a, b) for a, b in zip(s.smem_index, s.lane_index))
                if s.lane_index
                else s.smem_index
            )
            a, _ = self.addr(s.smem, indices)
            b, _ = self.addr(s.src, s.src_index)
            self.emit(f"cp.async.ca.shared.global [{a}], [{b}], {s.nbytes};")
        elif n == "CpAsyncCommit":
            self.emit("cp.async.commit_group;")
        elif n == "CpAsyncWait":
            self.emit(f"cp.async.wait_group {s.group};")
        elif n == "Sync":
            if s.warp:
                self.emit("bar.warp.sync 0xffffffff;")
            elif s.count is not None:
                self.emit(f"bar.sync {s.barrier_id}, {s.count};")
            else:
                self.emit(f"bar.sync {s.barrier_id};")
        elif n == "TreeHalve":
            # Each warp's lane0 has already stored its partial. Thread0 reduces these
            # using the IR's combine expressions, then broadcasts through shared memory.
            tid = self.mov("%tid.x")
            p = self.reg("p")
            self.emit(f"setp.ne.u32 {p}, {tid}, 0;")
            skip = self.label()
            self.emit(f"@{p} bra {skip};")
            for a, b, buf in zip(s.state, s.state_b, s.bufs):
                addr, _ = self.addr(buf, (0,))
                self.emit(f"ld.shared.f32 {self.env[a]}, [{addr}];")
            for i in range(1, s.length):
                for a, b, buf in zip(s.state, s.state_b, s.bufs):
                    addr, _ = self.addr(buf, (i,))
                    tmp = self.reg("f")
                    self.emit(f"ld.shared.f32 {tmp}, [{addr}];")
                    self.set(b, tmp)
                self.walk(s.combine_states)
            for a, buf in zip(s.state, s.bufs):
                addr, _ = self.addr(buf, (0,))
                self.emit(f"st.shared.f32 [{addr}], {self.env[a]};")
            self.emit(skip + ":")
            self.emit("bar.sync 0;")
            for a, buf in zip(s.state, s.bufs):
                addr, _ = self.addr(buf, (0,))
                self.emit(f"ld.shared.f32 {self.env[a]}, [{addr}];")
        else:
            raise NotImplementedError(("statement", n))

    def compile(self):
        self.accums = set()
        self.explicit_inits = set()

        def collect(body):
            for s in body:
                if type(s).__name__ == "Init":
                    self.explicit_inits.add(s.name)
                if type(s).__name__ == "Accum":
                    self.accums.add(s.name)
                if type(s).__name__ in ["WarpShuffle", "TreeHalve"]:
                    self.accums.update(s.state)
                for a in ["body", "else_body"]:
                    if hasattr(s, a):
                        collect(getattr(s, a))

        collect(self.k.body)
        for name in self.meta["arg_order"]:
            r = self.reg("rd")
            self.emit(f"ld.param.u64 {r}, [param_{name}];")
            self.ptr[name] = r
        self.walk(self.k.body)
        self.emit("ret;")
        decl = [
            ".version 8.0",
            ".target sm_89",
            ".address_size 64",
            "",
            f".visible .entry {self.k.name}(",
            ",\n".join(" .param .u64 param_" + n for n in self.meta["arg_order"]),
            ") {",
        ]
        for t, pt in [("r", "b32"), ("rd", "b64"), ("f", "f32"), ("p", "pred")]:
            decl.append(f" .reg .{pt} %{t}<{self.count[t]+1}>;")
        if self.meta.get("smem_bytes", 0) > 49152:
            decl.insert(3, ".extern .shared .align 16 .b8 _dynamic_smem[];")
        else:
            for name, shape in self.shared.items():
                decl.append(f" .shared .align 16 .b8 {name}[{math.prod(shape)*4}];")
        return "\n".join(decl + self.lines + ["}"])


def main(path):
    path = Path(path)
    g = Graph.from_dict(json.load(open(path)))
    meta = json.load(open(path.parent / "meta.json"))
    out = []
    for n in g.nodes.values():
        if type(n.op).__name__ == "KernelOp":
            m = next(m for m in meta["kernels"] if m["entry"] == n.op.name)
            out.append(Compiler(g, n.op, m).compile())
    return "\n".join(out)


if __name__ == "__main__":
    print(main(sys.argv[1]))
