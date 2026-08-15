import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


TASK_ORDER = [
    "minerva_math_algebra",
    "minerva_math_counting_and_prob",
    "minerva_math_geometry",
    "minerva_math_intermediate_algebra",
    "minerva_math_num_theory",
    "minerva_math_prealgebra",
    "minerva_math_precalc",
]

# Default offline/cache env for Hugging Face access in server runs.
# Use setdefault so CLI-provided env vars still take precedence.
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/root/.cache/huggingface/hub")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def trim_to_next_problem(text: str) -> str:
    if not text:
        return text
    m = re.search(r"(?is)\n\s*problem\s*:", text)
    if m is None:
        return text.strip()
    return text[: m.start()].strip()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_resp_text(sample: Dict[str, Any]) -> str:
    resps = sample.get("resps", [])
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    filtered = sample.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        first = filtered[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    return ""


def pick_samples_file(samples_dir: Path, task_name: str) -> Optional[Path]:
    pattern = str(samples_dir / f"samples_{task_name}_*.jsonl")
    files = [Path(p) for p in glob.glob(pattern)]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def load_task_dict(task_names: List[str]):
    # Compatible with multiple lm-eval versions
    try:
        from lm_eval.tasks import get_task_dict

        try:
            return get_task_dict(task_names)
        except TypeError:
            from lm_eval.tasks import TaskManager

            tm = TaskManager()
            return get_task_dict(task_names, task_manager=tm)
    except Exception:
        from lm_eval.tasks import TaskManager

        tm = TaskManager()
        loaded = tm.load_task_or_group(task_names)
        if isinstance(loaded, dict):
            return loaded
        raise RuntimeError("Failed to load lm_eval tasks for minerva_math subtasks.")


def pick_metric_value(metric_dict: Dict[str, Any], metric_name: str) -> Optional[float]:
    # process_results may return exact key or key with filter suffix.
    if metric_name in metric_dict and isinstance(metric_dict[metric_name], (int, float)):
        return float(metric_dict[metric_name])
    for k, v in metric_dict.items():
        if k.startswith(metric_name) and isinstance(v, (int, float)):
            return float(v)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate MATH log_samples with lm_eval process_results after trimming repeated Problem tails."
    )
    parser.add_argument(
        "samples_dir",
        type=str,
        help="Directory containing samples_minerva_math_*.jsonl and optionally results_*.json",
    )
    parser.add_argument(
        "--write-cleaned",
        action="store_true",
        help="Write cleaned per-sample outputs to <samples_dir>/samples_minerva_math_trimmed.cleaned.jsonl",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="",
        help="Optional path to save summary json. Default: <samples_dir>/postprocess_math_logsamples_lmeval.summary.json",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir).expanduser()
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples_dir not found: {samples_dir}")

    task_dict = load_task_dict(TASK_ORDER)

    subgroup_stats: Dict[str, Dict[str, float]] = {}
    cleaned_rows: List[Dict[str, Any]] = []

    total_exact: List[float] = []
    total_verify: List[float] = []
    total_count = 0
    used_files: Dict[str, str] = {}
    missing_tasks: List[str] = []

    for task_name in TASK_ORDER:
        sample_file = pick_samples_file(samples_dir, task_name)
        if sample_file is None:
            missing_tasks.append(task_name)
            continue
        used_files[task_name] = str(sample_file)

        rows = read_jsonl(str(sample_file))
        task = task_dict[task_name]
        exact_vals: List[float] = []
        verify_vals: List[float] = []

        for i, sample in enumerate(rows):
            doc = sample.get("doc", {})
            if not isinstance(doc, dict):
                continue

            raw_resp = extract_resp_text(sample)
            trimmed_resp = trim_to_next_problem(raw_resp)
            metrics = task.process_results(doc, [trimmed_resp])
            if not isinstance(metrics, dict):
                continue

            ex = pick_metric_value(metrics, "exact_match")
            mv = pick_metric_value(metrics, "math_verify")
            if ex is None:
                ex = 0.0
            if mv is None:
                mv = 0.0
            exact_vals.append(ex)
            verify_vals.append(mv)
            total_exact.append(ex)
            total_verify.append(mv)

            if args.write_cleaned:
                cleaned_rows.append(
                    {
                        "task": task_name,
                        "idx_in_task": i,
                        "doc_id": sample.get("doc_id"),
                        "raw_response": raw_resp,
                        "trimmed_response": trimmed_resp,
                        "exact_match": ex,
                        "math_verify": mv,
                    }
                )

        n = len(exact_vals)
        total_count += n
        subgroup_stats[task_name] = {
            "num_samples": float(n),
            "exact_match": (sum(exact_vals) / n) if n else 0.0,
            "math_verify": (sum(verify_vals) / n) if n else 0.0,
        }

    overall_exact = (sum(total_exact) / len(total_exact)) if total_exact else 0.0
    overall_verify = (sum(total_verify) / len(total_verify)) if total_verify else 0.0

    summary = {
        "num_fewshot": 4,
        "num_samples": total_count,
        "used_files": used_files,
        "missing_tasks": missing_tasks,
        "postprocess_lm_eval": {
            "exact_match": overall_exact,
            "math_verify": overall_verify,
        },
        "per_subgroup": subgroup_stats,
    }

    summary_out = (
        Path(args.summary_out).expanduser()
        if args.summary_out.strip()
        else (samples_dir / "postprocess_math_logsamples_lmeval.summary.json")
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_file"] = str(summary_out)

    print(json.dumps(summary, ensure_ascii=False))

    if args.write_cleaned:
        out = samples_dir / "samples_minerva_math_trimmed.cleaned.jsonl"
        write_jsonl(str(out), cleaned_rows)


if __name__ == "__main__":
    main()
