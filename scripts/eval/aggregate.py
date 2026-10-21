"""Per-rung mean / AUC_log / scene-paired delta tables from evaluation outputs.

    python scripts/eval/aggregate.py [ROOT] [--ref NAME] [--names A,B,...]
                                     [--datasets acid,re10k] [--views 2,8] [--csv OUT]

Reads ROOT/<experiment>/<dataset>_ctx<N>/metrics.json (ROOT defaults to
analysis_outputs; one row per scene, row["budget_metrics"][<total budget>][metric]
a list over the scene's target views) and prints, per (N, dataset) and
metric, one line per experiment:

  per-rung columns  dataset mean at each per-view budget (total / N): the mean
                    over scenes of the scene's mean over its target views
  AUC_log           trapezoid over the log2 budget axis normalised to [0, 1];
                    on a doubling grid of five rungs that is
                    (0.5*r1 + r2 + r3 + r4 + 0.5*r5) / 4, so every doubling
                    weighs the same
  dAUC, win%, n     scene-paired against --ref: mean over the shared scenes of
                    the per-scene AUC difference, the share of those scenes
                    where the experiment is better, and how many are shared

Only the rungs every listed experiment has are used, and a scene enters the
AUC only when it has all of them.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRICS = (("psnr", "higher"), ("lpips", "lower"), ("ssim", "higher"))


def load(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    """metrics.json -> {scene: {total budget: {metric: scene mean}}}."""
    out = {}
    for index, row in enumerate(json.loads(path.read_text())):
        scene = row.get("scene") or row.get("key") or str(index)
        if isinstance(scene, list):
            scene = scene[0]
        out[scene] = {
            int(budget): {
                metric: statistics.fmean(value) if isinstance(value, list) else value
                for metric, value in per_budget.items()
                if isinstance(value, (list, int, float)) and not isinstance(value, bool)
            }
            for budget, per_budget in row.get("budget_metrics", {}).items()
        }
    return out


def auc_log(budgets: list[int], values: list[float]) -> float:
    x = [math.log2(budget) for budget in budgets]
    if len(x) < 2:
        return values[0]
    return sum((x[i + 1] - x[i]) * (values[i] + values[i + 1]) / 2 for i in range(len(x) - 1)) / (x[-1] - x[0])


def scene_auc(data, budgets: list[int], metric: str) -> dict[str, float]:
    return {
        scene: auc_log(budgets, [by_budget[b][metric] for b in budgets])
        for scene, by_budget in data.items()
        if all(b in by_budget and metric in by_budget[b] for b in budgets)
    }


def collect(root: Path, names, datasets, views) -> dict[tuple[int, str], dict[str, dict]]:
    """-> {(N, dataset): {experiment: loaded metrics}}."""
    cells: dict[tuple[int, str], dict[str, dict]] = {}
    for path in sorted(root.glob("*/*_ctx*/metrics.json")):
        experiment = path.parent.parent.name
        dataset, _, num_views = path.parent.name.rpartition("_ctx")
        if not num_views.isdigit():
            continue
        if (names and experiment not in names) or (datasets and dataset not in datasets) \
                or (views and int(num_views) not in views):
            continue
        data = load(path)
        if data:
            cells.setdefault((int(num_views), dataset), {})[experiment] = data
    return cells


def table(num_views: int, dataset: str, runs: dict[str, dict], ref: str | None, records: list) -> None:
    budgets = sorted(set.intersection(*[
        set.intersection(*[set(by_budget) for by_budget in data.values()]) for data in runs.values()
    ]))
    if not budgets:
        print(f"\n### N={num_views} {dataset}: no budget rung shared by all experiments, skipped")
        return
    scenes = sorted({len(data) for data in runs.values()})
    print(f"\n### N={num_views} {dataset}: scenes per experiment {scenes}, "
          f"per-view budgets {[b // num_views for b in budgets]}")
    width = max(12, *(len(name) for name in runs)) + 2
    for metric, direction in METRICS:
        fmt = "{:9.3f}" if metric == "psnr" else "{:9.4f}"
        paired = f", dAUC scene-paired vs {ref}" if ref in runs else ""
        print(f"\n{metric.upper()} ({direction} is better{paired})")
        print(f"{'experiment':<{width}}" + "".join(f"{b // num_views:>9}" for b in budgets)
              + f"{'AUC_log':>10}" + (f"{'dAUC':>10}{'win%':>7}{'n':>6}" if ref in runs else ""))
        ref_auc = scene_auc(runs[ref], budgets, metric) if ref in runs else {}
        rows = []
        for name, data in runs.items():
            per_rung = [statistics.fmean(s[b][metric] for s in data.values() if b in s and metric in s[b])
                        for b in budgets]
            aucs = scene_auc(data, budgets, metric)
            shared = sorted(set(aucs) & set(ref_auc))
            deltas = [aucs[s] - ref_auc[s] for s in shared]
            delta = statistics.fmean(deltas) if deltas else float("nan")
            wins = sum((d > 0) == (direction == "higher") for d in deltas) / len(deltas) if deltas else float("nan")
            auc = statistics.fmean(aucs.values()) if aucs else float("nan")
            rows.append((name, per_rung, auc, delta, wins, len(shared)))
            records.append({"num_views": num_views, "dataset": dataset, "experiment": name, "metric": metric,
                            "scenes": len(aucs), "auc_log": auc, "d_auc": delta, "win": wins, "paired": len(shared),
                            **{f"pv{b // num_views}": v for b, v in zip(budgets, per_rung)}})
        for name, per_rung, auc, delta, wins, n in sorted(rows, key=lambda r: -r[2] if direction == "higher" else r[2]):
            line = f"{name:<{width}}" + "".join(fmt.format(v) for v in per_rung) + fmt.format(auc).rjust(10)
            if ref in runs:
                line += ("{:+10.3f}" if metric == "psnr" else "{:+10.4f}").format(delta) + f"{100 * wins:>6.0f}%{n:>6}"
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default="analysis_outputs")
    parser.add_argument("--ref", help="experiment the scene-paired deltas are taken against")
    parser.add_argument("--names", help="comma list of experiments (default: all found)")
    parser.add_argument("--datasets", help="comma list, e.g. acid,re10k (default: all found)")
    parser.add_argument("--views", help="comma list of context-view counts (default: all found)")
    parser.add_argument("--csv", help="also write every row to this CSV file")
    args = parser.parse_args()
    split = lambda value: [v for v in value.split(",") if v] if value else []
    cells = collect(Path(args.root), set(split(args.names)), set(split(args.datasets)),
                    {int(v) for v in split(args.views)})
    if not cells:
        raise SystemExit(f"no <experiment>/<dataset>_ctx<N>/metrics.json under {args.root}")
    records: list[dict] = []
    for (num_views, dataset), runs in sorted(cells.items()):
        table(num_views, dataset, runs, args.ref, records)
    if args.csv:
        keys = sorted({k for r in records for k in r}, key=lambda k: (k.startswith("pv"), int(k[2:]) if k.startswith("pv") else 0, k))
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
