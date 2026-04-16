#!/usr/bin/env python
"""Inspect all ogPASS pass_sampler.pt checkpoints to determine if the
PASS sampler ever learned meaningful importance weights.

Reports:
  - as_ raw + softmax values (mixing weights between importance/uniform)
  - Ws projection matrix norms
  - type_embeddings norms per type

Output: stdout + debug_runs/sampler_diagnostic.md
"""
from __future__ import annotations
import math
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "debug_runs" / "results"
OUTPUT = REPO / "debug_runs" / "sampler_diagnostic.md"


def find_checkpoints() -> list[Path]:
    """Find all pass_sampler.pt files under results/ogpass_*."""
    return sorted(RESULTS.glob("ogpass_*/rel-f1/*/pass_sampler.pt"))


def inspect(ckpt_path: Path) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    as_raw = state["as_"]
    as_soft = F.softmax(as_raw, dim=0)

    Ws = state["Ws"]
    ws_norm = Ws.norm().item()
    ws_min = Ws.min().item()
    ws_max = Ws.max().item()
    ws_std = Ws.std().item()

    type_emb = state["type_embeddings.weight"]
    type_norms = type_emb.norm(dim=1).tolist()

    # Relative name: ogpass_s0/rel-f1/driver-top3
    rel = ckpt_path.parent.relative_to(RESULTS)

    return {
        "name": str(rel),
        "as_raw": as_raw.tolist(),
        "as_soft": as_soft.tolist(),
        "as_imp_weight": as_soft[0].item(),
        "ws_norm": ws_norm,
        "ws_min": ws_min,
        "ws_max": ws_max,
        "ws_std": ws_std,
        "type_norms": type_norms,
        "num_types": len(type_norms),
    }


def format_report(records: list[dict]) -> str:
    lines = [
        "# PASS Sampler Diagnostic Report",
        "",
        f"Inspected **{len(records)}** checkpoints.",
        "",
        "## Summary",
        "",
        "Init values: `as_ = [0.5, 0.5]` -> softmax `[0.5, 0.5]`, "
        "`Ws` zero-init, `type_embeddings` random ~N(0,1).",
        "",
        "If `as_` softmax importance weight is near 0.50 and `Ws` L2 norm "
        "is small, the sampler never learned to prefer importance over uniform.",
        "",
    ]

    # Aggregate stats
    imp_weights = [r["as_imp_weight"] for r in records]
    ws_norms = [r["ws_norm"] for r in records]
    mean_imp = sum(imp_weights) / len(imp_weights)
    mean_ws = sum(ws_norms) / len(ws_norms)
    max_ws = max(ws_norms)
    min_ws = min(ws_norms)

    lines += [
        "### Aggregate",
        "",
        f"- Mean importance weight (softmax as_[0]): **{mean_imp:.4f}** "
        f"(init = 0.5000)",
        f"- Ws L2 norm: mean={mean_ws:.4f}, min={min_ws:.4f}, max={max_ws:.4f}",
        "",
    ]

    # Group by task
    groups: dict[str, list[dict]] = {}
    for r in records:
        parts = r["name"].split("/")
        task = parts[1] + "/" + parts[2] if len(parts) >= 3 else "unknown"
        groups.setdefault(task, []).append(r)

    for task, recs in sorted(groups.items()):
        lines += [
            f"## Task: `{task}` ({len(recs)} checkpoints)",
            "",
            "| checkpoint | as_raw | softmax(as_) | imp_w | Ws L2 | Ws std | Ws range |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in recs:
            name_short = r["name"].split("/")[0]
            as_r = f"[{r['as_raw'][0]:.4f}, {r['as_raw'][1]:.4f}]"
            as_s = f"[{r['as_soft'][0]:.4f}, {r['as_soft'][1]:.4f}]"
            lines.append(
                f"| {name_short} | {as_r} | {as_s} | "
                f"{r['as_imp_weight']:.4f} | {r['ws_norm']:.4f} | "
                f"{r['ws_std']:.6f} | [{r['ws_min']:.4f}, {r['ws_max']:.4f}] |"
            )
        lines.append("")

        # Type embedding norms for first checkpoint in group
        r0 = recs[0]
        lines += [
            f"Type embedding norms (from `{r0['name'].split('/')[0]}`):",
            "",
        ]
        for i, n in enumerate(r0["type_norms"]):
            lines.append(f"  - type {i}: {n:.4f}")
        lines.append("")

    # Verdict
    near_init = sum(1 for w in imp_weights if abs(w - 0.5) < 0.02)
    small_ws = sum(1 for n in ws_norms if n < 1.0)
    lines += [
        "## Verdict",
        "",
        f"- {near_init}/{len(records)} checkpoints have importance weight "
        f"within 0.02 of init (0.50)",
        f"- {small_ws}/{len(records)} checkpoints have Ws L2 norm < 1.0",
        "",
    ]
    if near_init > len(records) * 0.8 and small_ws > len(records) * 0.8:
        lines.append(
            "**Conclusion:** The sampler never learned meaningful importance "
            "weights. Proceed to Phase 2 fixes (REINFORCE baseline + AP tuning)."
        )
    else:
        lines.append(
            "**Conclusion:** Some checkpoints show movement from init. "
            "Investigate further before applying fixes."
        )
    lines.append("")

    return "\n".join(lines)


def main():
    ckpts = find_checkpoints()
    if not ckpts:
        print("No checkpoints found!")
        return

    print(f"Found {len(ckpts)} checkpoints\n")

    records = []
    for p in ckpts:
        try:
            r = inspect(p)
            records.append(r)
            print(
                f"  {r['name']:50s}  imp_w={r['as_imp_weight']:.4f}  "
                f"Ws_norm={r['ws_norm']:.6f}"
            )
        except Exception as e:
            print(f"  SKIP {p}: {e}")

    report = format_report(records)
    print("\n" + "=" * 60)
    print(report)

    OUTPUT.write_text(report)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
