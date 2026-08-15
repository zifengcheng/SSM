import argparse
import json
import os
from typing import Any, Dict, List, Optional

from sanitize import sanitize


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _extract_response_text(sample: Dict[str, Any]) -> str:
    resps = sample.get("resps", [])
    if not resps:
        return ""
    first = resps[0]
    if isinstance(first, list) and first:
        return str(first[0])
    return str(first)


def _strip_markdown_code_fence(text: str) -> str:
    if "```python\n" in text:
        return text.split("```python\n", 1)[-1].split("```", 1)[0]
    if "```" in text:
        return text.split("```", 1)[-1].split("```", 1)[0]
    return text


def _trim_mbpp_completion(text: str) -> str:
    """
    Keep only the first completion block for MBPP.
    Generated samples in this repo may continue with repeated few-shot prompts
    after the first solution; those tails should be dropped before code_eval.
    """
    if not text:
        return text

    # Primary cut: task template delimiter.
    done_pos = text.find("[DONE]")
    if done_pos != -1:
        text = text[:done_pos]

    # Fallback cut: repeated prompt prefix that should never appear in completion.
    marker = "\nYou are an expert Python programmer"
    marker_pos = text.find(marker)
    if marker_pos != -1:
        text = text[:marker_pos]

    # Fallback cut: repeated in-context delimiter.
    begin_pos = text.find("\n[BEGIN]")
    if begin_pos != -1:
        text = text[:begin_pos]

    return text.strip()


def _build_completion(sample: Dict[str, Any]) -> str:
    raw_resp = _extract_response_text(sample)
    candidate = _strip_markdown_code_fence(raw_resp)
    candidate = _trim_mbpp_completion(candidate)
    doc = sample.get("doc", {})
    prompt = str(doc.get("prompt", ""))
    entry_point = doc.get("entry_point")
    if entry_point is None:
        return candidate
    return sanitize(prompt + "\n" + candidate, entry_point)


def _existing_pass_at_1(sample: Dict[str, Any]) -> Optional[float]:
    value = sample.get("pass@1", None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _compute_pass_at_1_with_code_eval(
    references: List[str], predictions: List[List[str]]
) -> List[float]:
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    import evaluate as hf_evaluate

    pass_at_k = hf_evaluate.load("code_eval")
    scores = []
    for reference, prediction in zip(references, predictions):
        score = pass_at_k.compute(
            references=[reference],
            predictions=[prediction],
            k=[1],
        )[0]["pass@1"]
        scores.append(float(score))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess MBPP samples and compute Pass@1.")
    parser.add_argument("file_path", type=str, help="Path to lm-eval samples jsonl.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing `pass@1` field and recompute with `code_eval`.",
    )
    parser.add_argument(
        "--write-cleaned",
        action="store_true",
        help="Write per-sample cleaned outputs to `<file_path>.cleaned`.",
    )
    args = parser.parse_args()

    data = read_jsonl(args.file_path)
    if not data:
        raise ValueError(f"No samples found in {args.file_path}")

    existing_scores = [_existing_pass_at_1(sample) for sample in data]
    can_use_existing = all(score is not None for score in existing_scores)

    predictions = [[_build_completion(sample)] for sample in data]
    references = [str(sample.get("target", "")) for sample in data]

    if can_use_existing and not args.force_recompute:
        pass_at_1_scores = [float(score) for score in existing_scores if score is not None]
        score_source = "existing-pass@1"
    else:
        pass_at_1_scores = _compute_pass_at_1_with_code_eval(references, predictions)
        score_source = "recomputed-code_eval"

    avg_pass_at_1 = sum(pass_at_1_scores) / len(pass_at_1_scores)
    print(avg_pass_at_1)
    print(
        json.dumps(
            {
                "num_samples": len(data),
                "avg_pass@1": avg_pass_at_1,
                "score_source": score_source,
            },
            ensure_ascii=False,
        )
    )

    if args.write_cleaned:
        cleaned = [
            {
                "task_id": sample.get("doc", {}).get("task_id", sample.get("doc_id")),
                "completion": pred[0],
                "pass_at_1": score,
            }
            for sample, pred, score in zip(data, predictions, pass_at_1_scores)
        ]
        write_jsonl(cleaned, args.file_path + ".cleaned")


if __name__ == "__main__":
    main()
