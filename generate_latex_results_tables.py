#!/usr/bin/env python3
"""Generate LaTeX metric tables from recovered completed evaluations.

The all-results table contains final logged accuracy, macro recall and macro F1
for every completed optimizer.  The wins table contains the best learned
optimizer for each experiment where its final macro F1 strictly exceeds the
best classical baseline.  Multiple completed seeds are averaged.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


METRICS = ("accuracy", "recall", "f1")


def latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def task_name(payload, path):
    source = payload.get("recovery", {}).get("source_log")
    name = Path(source).name if source else path.stem
    if name.startswith("output_"):
        name = name[len("output_"):]
    if name.endswith("_recovered"):
        name = name[:-len("_recovered")]
    return name


def is_learned_optimizer(name):
    lowered = name.lower()
    return "gnn" in lowered or "lstm-dm" in lowered or "learned" in lowered


def final_seed_metrics(step_values):
    candidates = []
    for step, values in step_values.items():
        if not isinstance(values, dict):
            continue
        if all(values.get(metric) is not None for metric in METRICS):
            try:
                candidates.append((int(step), values))
            except (TypeError, ValueError):
                pass
    if not candidates:
        return None
    step, values = max(candidates, key=lambda item: item[0])
    return step, {metric: float(values[metric]) for metric in METRICS}


def collect_rows(input_root):
    grouped = defaultdict(lambda: defaultdict(list))
    sources = {}
    for path in sorted(input_root.rglob("*_recovered.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print("warning: could not read {}: {}".format(path, error))
            continue
        task = task_name(payload, path)
        source_format = payload.get("recovery", {}).get("format", "unknown")
        classification = payload.get("classification_metrics", {})
        if not isinstance(classification, dict):
            continue
        for problem, optimizers in classification.items():
            if not isinstance(optimizers, dict):
                continue
            experiment = (task, str(problem), source_format)
            sources[experiment] = str(path)
            for optimizer, seeds in optimizers.items():
                if not isinstance(seeds, dict):
                    continue
                for seed, step_values in seeds.items():
                    if not isinstance(step_values, dict):
                        continue
                    endpoint = final_seed_metrics(step_values)
                    if endpoint is not None:
                        step, metrics = endpoint
                        grouped[experiment][str(optimizer)].append((str(seed), step, metrics))

    rows = []
    for experiment, optimizers in sorted(grouped.items()):
        for optimizer, seed_values in sorted(optimizers.items()):
            rows.append({
                "task": experiment[0],
                "problem": experiment[1],
                "format": experiment[2],
                "optimizer": optimizer,
                "learned": is_learned_optimizer(optimizer),
                "seeds": len(seed_values),
                "step": min(value[1] for value in seed_values),
                "accuracy": sum(value[2]["accuracy"] for value in seed_values) / len(seed_values),
                "recall": sum(value[2]["recall"] for value in seed_values) / len(seed_values),
                "f1": sum(value[2]["f1"] for value in seed_values) / len(seed_values),
                "source": sources[experiment],
            })
    return rows


def metric_text(value, bold=False):
    rendered = "{:.3f}".format(value)
    return r"\textbf{" + rendered + "}" if bold else rendered


def write_all_results(rows, destination):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["problem"], row["format"])].append(row)

    lines = [
        r"% Requires \usepackage{booktabs,longtable}",
        r"\begin{longtable}{lllrccc}",
        r"\caption{Final classification metrics for all fully evaluated optimizers.}\label{tab:all-results}\\",
        r"\toprule",
        r"Task & Problem & Optimizer & Seeds & Accuracy & Recall & Macro-F1 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Task & Problem & Optimizer & Seeds & Accuracy & Recall & Macro-F1 \\",
        r"\midrule",
        r"\endhead",
    ]
    for experiment in sorted(grouped):
        experiment_rows = grouped[experiment]
        maxima = {metric: max(row[metric] for row in experiment_rows) for metric in METRICS}
        for row in sorted(experiment_rows, key=lambda item: (-item["f1"], item["optimizer"])):
            lines.append("{} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["task"]),
                latex_escape(row["problem"]),
                latex_escape(row["optimizer"]),
                row["seeds"],
                metric_text(row["accuracy"], row["accuracy"] == maxima["accuracy"]),
                metric_text(row["recall"], row["recall"] == maxima["recall"]),
                metric_text(row["f1"], row["f1"] == maxima["f1"]),
            ))
        lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


def winning_rows(rows, minimum_margin):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["problem"], row["format"])].append(row)
    wins = []
    for experiment, experiment_rows in sorted(grouped.items()):
        learned = [row for row in experiment_rows if row["learned"]]
        classical = [row for row in experiment_rows if not row["learned"]]
        if not learned or not classical:
            continue
        best_learned = max(learned, key=lambda row: row["f1"])
        best_classical = max(classical, key=lambda row: row["f1"])
        margin = best_learned["f1"] - best_classical["f1"]
        if margin > minimum_margin:
            winner = dict(best_learned)
            winner["baseline"] = best_classical["optimizer"]
            winner["baseline_f1"] = best_classical["f1"]
            winner["f1_margin"] = margin
            wins.append(winner)
    return wins


def write_wins(rows, destination):
    lines = [
        r"% Requires \usepackage{booktabs,longtable}",
        r"\begin{longtable}{lllcccc}",
        r"\caption{Experiments where a learned optimizer has higher final macro-F1 than every classical baseline.}\label{tab:learned-wins}\\",
        r"\toprule",
        r"Task & Learned optimizer & Best baseline & Accuracy & Recall & Macro-F1 & $\Delta$F1 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Task & Learned optimizer & Best baseline & Accuracy & Recall & Macro-F1 & $\Delta$F1 \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in sorted(rows, key=lambda item: (-item["f1_margin"], item["task"])):
        task = row["task"] if row["task"] == row["problem"] else "{} / {}".format(
            row["task"], row["problem"]
        )
        lines.append("{} & {} & {} & {:.3f} & {:.3f} & {:.3f} & {:+.3f} \\\\".format(
            latex_escape(task),
            latex_escape(row["optimizer"]),
            latex_escape(row["baseline"]),
            row["accuracy"], row["recall"], row["f1"], row["f1_margin"],
        ))
    if not rows:
        lines.append(r"\multicolumn{7}{c}{No learned-optimizer wins found.} \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


def atomic_write_tables(rows, wins, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        (output_dir / "all_results.tex", lambda path: write_all_results(rows, path)),
        (output_dir / "learned_optimizer_wins.tex", lambda path: write_wins(wins, path)),
    ]
    for destination, writer in targets:
        temporary = destination.with_name(destination.name + ".tmp")
        writer(temporary)
        os.replace(str(temporary), str(destination))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="recovered_partial_results")
    parser.add_argument("--output-dir", default="latex_tables")
    parser.add_argument(
        "--minimum-f1-margin",
        type=float,
        default=0.0,
        help="Required learned-minus-classical final macro-F1 margin (default: strict win).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    rows = collect_rows(Path(args.input_dir).resolve())
    wins = winning_rows(rows, args.minimum_f1_margin)
    output_dir = Path(args.output_dir).resolve()
    atomic_write_tables(rows, wins, output_dir)
    manifest = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "result_rows": len(rows),
        "learned_optimizer_wins": len(wins),
        "winning_rule": "best learned final macro-F1 > best classical final macro-F1",
        "minimum_f1_margin": args.minimum_f1_margin,
        "files": [str(output_dir / "all_results.tex"), str(output_dir / "learned_optimizer_wins.tex")],
    }
    (output_dir / "table_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote {} result rows and {} learned-optimizer wins to {}".format(
        len(rows), len(wins), output_dir
    ))


if __name__ == "__main__":
    main()
