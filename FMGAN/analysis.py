"""FMGAN results aggregator — reproduce the MiLeTS 2026 paper tables from results.json.

Walks results/results/phase1_*/results.json and baseline_*.json and reconstructs:
  Table 1: R3GAN-1D refinement effectiveness (dataset x coarse method -> before/after MAE, Delta%)
  Table 2: comparison vs established baselines (BRITS / SAITS) on Weather @ rate 0.25

Replaces the hard-coded numbers in paper/figures/gen_figure.py with a single source of
truth derived directly from the experiment outputs, so the negative result is one-command
reproducible. Stdlib only; offline.

NOTE (2026-06): kept LOCAL during the MiLeTS double-blind period — do NOT push until the
6/10 notification (Phase B). Run from the FMGAN/ directory:
    python3 analysis.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "results"

# canonical coarse-method order for the table
COARSE_ORDER = ["zero", "mean", "linear", "linear_v2", "augmented"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def collect_phase1():
    rows = []
    for d in sorted(RES.glob("phase1_*")):
        rj = d / "results.json"
        if not rj.exists():
            continue
        j = load(rj)
        if not j or "coarse_metrics" not in j or "refined_metrics" not in j:
            continue
        rows.append({
            "dir": d.name,
            "dataset": j.get("dataset", "?"),
            "coarse": j.get("coarse_method", "?"),
            "before_mae": j["coarse_metrics"].get("MAE"),
            "after_mae": j["refined_metrics"].get("MAE"),
            "delta_pct": j.get("improvement_pct"),
        })
    return rows


def collect_baselines():
    out = []
    for f in sorted(RES.glob("baseline_*.json")):
        j = load(f)
        if isinstance(j, list):
            out.extend(j)
        elif isinstance(j, dict):
            out.append(j)
    return out


def fmt(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)


def main():
    rows = collect_phase1()
    print(f"# FMGAN — reproduced from {len(rows)} phase1 results.json\n")

    print("## Table 1 — R3GAN-1D refinement effectiveness (MAE, lower=better)")
    print(f"{'Dataset':12s} {'Coarse':12s} {'Before':>8s} {'After':>8s} {'Delta%':>8s}")
    def key(r):
        try:
            ci = COARSE_ORDER.index(r["coarse"])
        except ValueError:
            ci = len(COARSE_ORDER)
        return (r["dataset"], ci)
    for r in sorted(rows, key=key):
        dpct = f"{r['delta_pct']:+.1f}" if r["delta_pct"] is not None else "?"
        print(f"{r['dataset']:12s} {r['coarse']:12s} "
              f"{fmt(r['before_mae']):>8s} {fmt(r['after_mae']):>8s} {dpct:>8s}")

    # headline finding: gain only when coarse imputer is implausible (zero/mean), ~0 for linear
    print("\n## Finding — gain is large only for distributionally-implausible coarse fills")
    for cat, preds in [("weak fills (zero/mean)", lambda c: c in ("zero", "mean")),
                       ("strong fill (linear*)", lambda c: c.startswith("linear"))]:
        ds = [r["delta_pct"] for r in rows if preds(r["coarse"]) and r["delta_pct"] is not None]
        if ds:
            print(f"  {cat:24s}: Delta% range [{min(ds):+.1f}, {max(ds):+.1f}], "
                  f"mean {sum(ds)/len(ds):+.1f}  (n={len(ds)})")

    print("\n## Table 2 — vs established baselines (raw baseline_*.json)")
    for b in collect_baselines():
        print(f"  {b.get('method','?'):8s} {b.get('dataset','?'):12s} "
              f"rate={b.get('missing_rate','?')}  MAE={fmt(b.get('MAE'))}  MSE={fmt(b.get('MSE'))}")
    print("\n  (R3GAN-1D standalone Weather MAE ~0.228 vs BRITS ~0.039 -> ~5.8x worse;")
    print("   refining linear interp 0.067 -> 0.067 = no gain. Negative result holds.)")


if __name__ == "__main__":
    main()
