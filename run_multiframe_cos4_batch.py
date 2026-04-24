import argparse
import csv
import json
from pathlib import Path

from multiframe_cos4_simple_benchmark import main as _unused_main  # keeps module dependency explicit


def run_case(case_dir: Path, output_root: Path, benchmark_summary: Path, stat: str, target: str, fuse: str) -> dict:
    import subprocess
    import sys

    case_manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    case_id = str(case_manifest["case_id"])
    case_out = output_root / case_id
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("multiframe_cos4_simple_benchmark.py")),
        "--case-dir",
        str(case_dir),
        "--output-root",
        str(case_out),
        "--benchmark-summary",
        str(benchmark_summary),
        "--stat",
        stat,
        "--target",
        target,
        "--fuse",
        fuse,
    ]
    subprocess.run(cmd, check=True)

    summary = json.loads((case_out / "summary.json").read_text(encoding="utf-8"))
    group_rows = json.loads((case_out / "comparison_against_group.json").read_text(encoding="utf-8"))["group_benchmark_reference"]
    return {
        "case_id": case_id,
        "display_name": str(case_manifest["display_name"]),
        "cos4_summary": summary,
        "group_rows": group_rows,
        "output_dir": str(case_out),
    }


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "case_id",
        "display_name",
        "model",
        "runtime_sec",
        "v_fit_percent",
        "v_r2",
        "v_rmse",
        "v_mae",
        "equalization_stat",
        "equalization_target",
        "fusion",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run multiframe simple cos4 benchmark on multiple prepared cases.")
    parser.add_argument(
        "--prepared-case-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Prepared dataset directories, each containing manifest.json and sim_capture_*.npy files.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-summary", type=Path, default=Path("real_benchmark_outputs") / "benchmark_summary.csv")
    parser.add_argument("--stat", type=str, default="p99.8", choices=["p99.8", "p95"])
    parser.add_argument("--target", type=str, default="unity", choices=["unity", "median"])
    parser.add_argument("--fuse", type=str, default="mean", choices=["mean", "median"])
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    aggregate_rows = []
    comparison_rows = []
    for case_dir in args.prepared_case_dirs:
        result = run_case(case_dir, args.output_root, args.benchmark_summary, args.stat, args.target, args.fuse)
        results.append(result)

        cos4_summary = result["cos4_summary"]
        aggregate_rows.append(
            {
                "case_id": result["case_id"],
                "display_name": result["display_name"],
                "model": cos4_summary["model"],
                "runtime_sec": cos4_summary["runtime_sec"],
                "v_fit_percent": cos4_summary["v_metrics"]["fit_percent"],
                "v_r2": cos4_summary["v_metrics"]["r2"],
                "v_rmse": cos4_summary["v_metrics"]["rmse"],
                "v_mae": cos4_summary["v_metrics"]["mae"],
                "equalization_stat": cos4_summary["equalization"]["stat"],
                "equalization_target": cos4_summary["equalization"]["target"],
                "fusion": cos4_summary["fusion"],
            }
        )

        comparison_rows.append(
            {
                "case_id": result["case_id"],
                "display_name": result["display_name"],
                "model": cos4_summary["model"],
                "v_fit_percent": cos4_summary["v_metrics"]["fit_percent"],
                "v_rmse": cos4_summary["v_metrics"]["rmse"],
            }
        )
        for row in result["group_rows"]:
            comparison_rows.append(
                {
                    "case_id": row["case_id"],
                    "display_name": row["display_name"],
                    "model": row["model"],
                    "v_fit_percent": float(row["v_fit_percent"]),
                    "v_rmse": float(row["v_rmse"]),
                }
            )

    write_rows_csv(args.output_root / "multiframe_cos4_batch_summary.csv", aggregate_rows)
    (args.output_root / "multiframe_cos4_batch_summary.json").write_text(
        json.dumps(
            [
                {
                    **row,
                    "equalization_target": float(row["equalization_target"])
                    if isinstance(row["equalization_target"], (int, float))
                    else row["equalization_target"],
                }
                for row in aggregate_rows
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_root / "comparison_against_existing.json").write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
