# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the SM120 rebase series: no SM100-only code, and the opted-in defaults.

Run it against the branch diff:

    python tools/sm120_scope_check.py origin/main

It checks four things the series promised:

1. Nothing added references a family-100-only facility (tcgen05 / MegaMoE /
   DeepSelect / gvr_2 / `is_device_capability_family(100)`), since none of it can
   run on the target hardware and the union with upstream must not import it.
2. The family-12 cache geometry keeps a 64-state page for every compression
   ratio, and the SWA page is 64 only on family 12.
3. The three correctness guards are present: the local->global top-k bound, the
   absent-parameter skip in the loaders, and the post-capture KV restore.
4. Every switch the series adds defaults to the conservative value, so serving
   behaviour is unchanged unless the operator opts in.
"""

from __future__ import annotations

import re
import subprocess
import sys

REPO = "."

# 1. Facilities that exist only for the datacenter Blackwell family.
SM100_ONLY = [
    r"is_device_capability_family\(100\)",
    r"\bsm_100",
    r"\btcgen05",
    r"deep_gemm_mega_moe",
    r"deep_select",
    r"gvr_2",
]

# 4. Switches added by this series and the default each must keep. `True` means
# the switch has to be inert by default, which for a bool env means "0". After
# the cleanup only the profiling aids and the MXFP8 policy survive: every other
# switch either became the only code path or was dropped.
CONSERVATIVE_DEFAULTS = {
    "VLLM_PROFILE_DIR": '""',
    "VLLM_GRAPH_KERNEL_ANNOTATIONS": "0",
    "VLLM_GEMM_SHAPE_LOG": "0",
    "VLLM_COMM_SHAPE_LOG": "0",
    "VLLM_MOE_PAD_LOG": "0",
    "VLLM_MEM_LOG": "0",
    "VLLM_KV_ACCOUNTING_LOG": "0",
    "VLLM_MXFP8_BACKEND_POLICY": '""',
}

# Names the code reads off `vllm.envs` (directly or through an alias) but that
# the module never declares: reading one raises AttributeError at the first
# serve, which is how VLLM_MOE_PAD_LOG took the engine down once.
ENV_READ = r"\b\w*envs\.(VLLM_[A-Z0-9_]+)"

GUARDS = {
    "top-k index bound": (
        "vllm/models/deepseek_v41/common/ops/cache_utils.py",
        r"local_idx < max_local_index",
    ),
    "absent checkpoint parameter": (
        "vllm/models/deepseek_v41/nvidia/model.py",
        r"Skipping (stacked )?checkpoint weight with no parameter",
    ),
    "post-capture KV restore": (
        "vllm/v1/worker/utils.py",
        r"def zero_captured_kv_caches",
    ),
    "family-12 page geometry": (
        "vllm/models/deepseek_v41/attention.py",
        r"def _sm12x_sparse_page_tokens",
    ),
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", REPO, *args], check=True, capture_output=True, text=True
    ).stdout


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    diff = _git("diff", f"{base}...HEAD")
    # Only added lines of Python sources under vllm/ count: docs and comments in
    # the plan file legitimately name the facilities this check forbids.
    added: list[str] = []
    current = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif (
            line.startswith("+")
            and not line.startswith("+++")
            and current.startswith("vllm/")
            and current.endswith(".py")
        ):
            added.append(line[1:])
    failures: list[str] = []

    for pattern in SM100_ONLY:
        hits = [line.strip() for line in added if re.search(pattern, line)]
        if hits:
            failures.append(
                f"SM100-only facility {pattern!r} introduced:\n    "
                + "\n    ".join(hits[:3])
            )

    for name, (path, pattern) in GUARDS.items():
        try:
            with open(path) as handle:
                source = handle.read()
        except OSError as exc:  # pragma: no cover - diagnostic path
            failures.append(f"{name}: cannot read {path}: {exc}")
            continue
        if not re.search(pattern, source):
            failures.append(f"{name}: missing in {path} (expected /{pattern}/)")

    # Collapse the source so multi-line lambdas match a single pattern.
    with open("vllm/envs.py") as handle:
        envs = re.sub(r"\s+", " ", handle.read())
    for name, expected in CONSERVATIVE_DEFAULTS.items():
        match = re.search(
            rf'"{name}": lambda: [^;]*?os\.getenv\("{name}", "?([^",)]*)"?\)',
            envs,
        )
        if match is None:
            failures.append(f"{name}: no default found in vllm/envs.py")
        elif match.group(1) != expected.strip('"'):
            failures.append(
                f"{name}: default is {match.group(1)!r}, expected {expected!r}"
            )

    # Every env var the code reads must be declared with a default.
    import pathlib

    declared = set(re.findall(r'"(VLLM_[A-Z0-9_]+)":', envs))
    used: set[str] = set()
    for path in pathlib.Path("vllm").rglob("*.py"):
        used |= set(re.findall(ENV_READ, path.read_text(errors="ignore")))
    for name in sorted(used - declared):
        failures.append(f"{name}: read from vllm.envs but not declared")

    if failures:
        print("sm120 scope check FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        "sm120 scope check passed: no SM100-only code, guards present, defaults inert"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
