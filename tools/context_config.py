#!/usr/bin/env python3
"""tools/context_config.py - a setup-written engine config at another context length, for the run-<model> scripts.

    python tools/context_config.py strata-iq2_xs.json 1M      -> prints the path of strata-iq2_xs.ctx1048576.json

CONTEXT is a token count (131072) or a count with a K / M suffix (512K, 1M; K = 1024).  The new config is the given
one with the context-dependent engine flags redone by setup.py's rules (its step 7):

  * --max-context CONTEXT;
  * rope scaling only past the trained 262,144 positions: yarn, factor CONTEXT / 262144 (an explicit --rope-scaling
    linear in the source config is kept as the method; its factor is derived again);
  * KV streaming (--kv-resident 32768) from 64K up only where setup would keep it: the source config's own choice
    when it is 64K or more (setup already weighed the RAM, WSL and --kv-streaming for it); from a shorter source,
    on when the KV cache's RAM fits beside the model's RAM use like setup's check (never with --kv k8v4 or on WSL);
  * --kv int8 above 8K when the source config has no --kv (an 8K install runs fp16).

Everything else (pack, model files, GPUs, port, log) is copied unchanged.  The source config is never modified.
"""
import json
import re
import sys
from pathlib import Path

TRAINED = 262144            # the model's trained context (setup.py resolve_rope)
STREAM_FROM = 65536         # setup.py: KV streaming from 64K
STREAM_CELLS = "32768"


def parse_context(text: str) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", text)
    if not m:
        raise ValueError(f"not a context length: {text!r} (use e.g. 131072, 512K or 1M)")
    n = float(m.group(1)) * {"": 1, "k": 1024, "m": 1024 * 1024}[m.group(2).lower()]
    if n != int(n) or n < 256:
        raise ValueError(f"context {text!r}: a whole number of tokens, 256 or more")
    return int(n)


def drop(args: list, flag: str, has_value: bool = True) -> list:
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2 if has_value else 1
            continue
        out.append(args[i])
        i += 1
    return out


def value(args: list, flag: str):
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def mem_gb():
    """MemTotal in GB (None where /proc/meminfo is missing)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        pass
    return None


def kv_ram_gb(ctx: int, kv) -> float:
    return ctx * 13 * (576 if kv == "q4_0" else 1056) / 1e9   # setup.py: 12 QSA layers + the draft layer


def model_ram_gb(src: Path) -> float:
    """setup.py MODELS' ram_gb of the model this config (strata-<tag>.json) is for; the largest when unknown."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from setup import FAMILIES, MODELS
    tag = src.stem.split(".")[0].removeprefix("strata-")
    for fam in FAMILIES.values():               # strata-coder-iq3_xxs.json: the family's tag, then the model
        if fam["tag"] and tag.startswith(fam["tag"]):
            tag = tag[len(fam["tag"]):]
    by_tag = {m.lower(): d["ram_gb"] for m, d in MODELS.items()}
    return by_tag.get(tag, max(by_tag.values()))


def streams(args: list, ctx: int, kv, src: Path) -> bool:
    if ctx < STREAM_FROM or kv == "k8v4" or is_wsl():
        return False
    src_ctx = int(value(args, "--max-context") or 0)
    if src_ctx >= STREAM_FROM:              # setup decided for this config: keep its choice
        return "--kv-resident" in args
    ram = mem_gb()                          # setup: the model's RAM use + the KV cache + 1 GB
    return ram is not None and ram >= model_ram_gb(src) + kv_ram_gb(ctx, kv) + 1


def with_context(args: list, ctx: int, src: Path) -> list:
    method = value(args, "--rope-scaling")
    kv = value(args, "--kv") or ("int8" if ctx > 8192 else None)
    stream = streams(args, ctx, kv, src)
    for flag in ("--max-context", "--rope-scaling", "--rope-scale", "--kv-resident"):
        args = drop(args, flag)
    args += ["--max-context", str(ctx)]
    if ctx > TRAINED:
        args += ["--rope-scaling", method if method in ("yarn", "linear") else "yarn",
                 "--rope-scale", f"{ctx / TRAINED:g}"]
    if kv is not None and value(args, "--kv") is None:
        args += ["--kv", kv]
    if stream:
        args += ["--kv-resident", STREAM_CELLS]
    return args


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    src = Path(sys.argv[1]).resolve()
    try:
        ctx = parse_context(sys.argv[2])
    except ValueError as e:
        print(f"context_config: {e}", file=sys.stderr)
        return 2
    cfg = json.loads(src.read_text(encoding="utf-8-sig"))
    cfg["args"] = with_context(list(cfg["args"]), ctx, src)
    dst = src.with_name(f"{src.stem}.ctx{ctx}.json")
    dst.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    kv_gb = kv_ram_gb(ctx, value(cfg["args"], "--kv"))
    rope = value(cfg["args"], "--rope-scaling")
    print(f"context_config: {ctx} tokens" + (f", rope {rope} x{value(cfg['args'], '--rope-scale')} (experimental)"
                                             if rope else "") + f", KV cache ~{kv_gb:.1f} GB"
          + (" (streamed from RAM)" if "--kv-resident" in cfg["args"] else ""), file=sys.stderr)
    print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
