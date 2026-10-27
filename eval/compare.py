"""Compare saved eval reports across embedding configs, pooling question sets.

    python -m eval.compare
    python -m eval.compare --configs text-embedding-3-small-1536,text-embedding-3-large-3072
    python -m eval.compare --sets questions_multilingual

Reads eval/results/latest-<set>-<config>.json (written by eval.run_eval) and prints
hit@1 / hit@5 / MRR per config and mode, overall and per tag. Pooling the sets gives
each tag more questions than any single set, so differences are less noisy.
"""
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CONFIGS = "text-embedding-3-small-1536,text-embedding-3-large-1536,text-embedding-3-large-3072"
DEFAULT_SETS = "questions,questions_holdout,questions_multilingual"

def metrics(records):
    n = len(records)
    ranks = [r["hit_rank"] for r in records]
    return {
        "n": n,
        "hit@1": sum(1 for x in ranks if x == 1) / n,
        "hit@5": sum(1 for x in ranks if x) / n,
        "mrr": sum(1 / x for x in ranks if x) / n,
    }

def table(rows, columns):
    widths = [max(len(c), *(len(str(r[i])) for r in rows)) for i, c in enumerate(columns)]
    line = lambda vals: "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    return "\n".join([line(columns), line("-" * w for w in widths), *(line(r) for r in rows)])

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--sets", default=DEFAULT_SETS)
    parser.add_argument("--modes", default="vector,hybrid")
    args = parser.parse_args()
    configs, sets, modes = args.configs.split(","), args.sets.split(","), args.modes.split(",")

    # records[(config, mode)] = per-question records pooled over all sets
    records = {}
    for config in configs:
        for set_name in sets:
            path = RESULTS_DIR / f"latest-{set_name}-{config}.json"
            if not path.exists():
                raise SystemExit(f"Missing {path.name}: run python -m eval.run_eval for it first")
            report = json.loads(path.read_text(encoding="utf-8"))
            for mode in modes:
                records.setdefault((config, mode), []).extend(report["per_question"][mode])

    pct = lambda x: f"{x * 100:.1f}%"
    first = records[(configs[0], modes[0])]
    print(f"Pooled sets: {', '.join(sets)} ({len(first)} questions)\n")

    print("Overall")
    print(table(
        [[c, m, pct(s["hit@1"]), pct(s["hit@5"]), f"{s['mrr']:.3f}"]
         for (c, m), recs in records.items() for s in [metrics(recs)]],
        ["config", "mode", "hit@1", "hit@5", "MRR"],
    ))

    tags = sorted({t for r in first for t in r["tags"]})
    for mode in modes:
        print(f"\nBy tag, {mode} (hit@1 / hit@5)")
        rows = []
        for tag in tags:
            cells = []
            for c in configs:
                s = metrics([r for r in records[(c, mode)] if tag in r["tags"]])
                cells.append(f"{pct(s['hit@1'])} / {pct(s['hit@5'])}")
            n = sum(1 for r in first if tag in r["tags"])
            rows.append([tag, str(n), *cells])
        print(table(rows, ["tag", "n", *configs]))

if __name__ == "__main__":
    main()
