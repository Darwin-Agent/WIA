#!/usr/bin/env python3
"""Summarize rollout eval JSONL files by checkpoint, split, and domain."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SCORE_NAMES = ("consistency", "efficiency", "factuality", "format", "quality", "realism")
ID_DOMAINS = ("android", "mcp", "os", "search", "swe", "terminal", "web")
OOD_DOMAINS = ("android", "mcp", "os", "search", "swe", "terminal", "web")
SPLIT_SPECS = (("ID", ID_DOMAINS), ("OOD", OOD_DOMAINS))


@dataclass
class Bucket:
    n: int = 0
    reward_sum: float = 0.0
    weighted_sum: float = 0.0
    rule_reward_sum: float = 0.0
    rule_reward_count: int = 0
    fallback_count: int = 0
    similarity_sum: float = 0.0
    similarity_count: int = 0
    score_sums: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in SCORE_NAMES})

    def add(self, obj: dict) -> None:
        self.n += 1
        self.reward_sum += as_float(obj.get("judge_reward", obj.get("reward")))
        self.weighted_sum += as_float(obj.get("judge_weighted_score"))
        rule_reward = obj.get("judge_rule_based_reward")
        if rule_reward is not None:
            self.rule_reward_sum += as_float(rule_reward)
            self.rule_reward_count += 1
        if obj.get("judge_fallback") is True or rule_reward is not None:
            self.fallback_count += 1
        similarity_score = obj.get("similarity_score")
        if similarity_score is not None:
            self.similarity_sum += as_float(similarity_score)
            self.similarity_count += 1
        scores = obj.get("judge_scores") or {}
        for name in SCORE_NAMES:
            self.score_sums[name] += as_float(obj.get(f"judge_{name}", scores.get(name)))

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.n if self.n else 0.0

    @property
    def mean_weighted(self) -> float:
        return self.weighted_sum / self.n if self.n else 0.0

    @property
    def mean_rule_reward(self) -> float:
        return self.rule_reward_sum / self.rule_reward_count if self.rule_reward_count else 0.0

    @property
    def mean_similarity(self) -> float:
        return self.similarity_sum / self.similarity_count if self.similarity_count else 0.0

    def mean_score(self, name: str) -> float:
        return self.score_sums[name] / self.n if self.n else 0.0


def as_float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def ckpt_from_path(path: Path) -> int:
    match = re.search(r"eval_test_(\d+)\.jsonl$", path.name)
    if not match:
        raise ValueError(f"cannot parse checkpoint from {path}")
    return int(match.group(1))


def split_and_domain(obj: dict) -> tuple[str, str]:
    metadata = obj.get("metadata") or {}
    source = str(metadata.get("source") or "")
    domain = str(metadata.get("domain") or "unknown")
    if source.startswith("ID/"):
        return "ID", domain
    if source.startswith("OOD/"):
        return "OOD", domain
    dataset = str(metadata.get("dataset") or "")
    if dataset in {"AWM", "A11y-CUA", "AndroidControl", "OpenResearcher", "SWE-Gym", "TMax", "A3-Synth", "swe__openhands_sft", "web__weblinx_browsergym"}:
        return "ID", domain
    return "OOD", domain


def read_eval_files(root: Path) -> dict[int, dict[tuple[str, str], Bucket]]:
    grouped: dict[int, dict[tuple[str, str], Bucket]] = {}
    for path in sorted(root.glob("eval_test_*.jsonl"), key=ckpt_from_path):
        ckpt = ckpt_from_path(path)
        buckets: dict[tuple[str, str], Bucket] = defaultdict(Bucket)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                buckets[split_and_domain(obj)].add(obj)
        grouped[ckpt] = dict(buckets)
    return grouped


def aggregate_split(buckets: dict[tuple[str, str], Bucket], split: str) -> Bucket:
    total = Bucket()
    for (item_split, _domain), bucket in buckets.items():
        if item_split != split:
            continue
        total.n += bucket.n
        total.reward_sum += bucket.reward_sum
        total.weighted_sum += bucket.weighted_sum
        for name in SCORE_NAMES:
            total.score_sums[name] += bucket.score_sums[name]
    return total


def ordered_rows(
    grouped: dict[int, dict[tuple[str, str], Bucket]], splits: tuple[str, ...] = ("ID", "OOD")
) -> Iterable[tuple[int, str, str, Bucket]]:
    for ckpt in sorted(grouped):
        for split, domains in SPLIT_SPECS:
            if split not in splits:
                continue
            for domain in domains:
                bucket = grouped[ckpt].get((split, domain))
                if bucket is not None:
                    yield ckpt, split, domain, bucket


def write_csvs(
    root: Path, grouped: dict[int, dict[tuple[str, str], Bucket]], splits: tuple[str, ...] = ("ID", "OOD")
) -> None:
    with (root / "eval_reward_by_ckpt_split_domain.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ckpt", "split", "domain", "n", "mean_reward"])
        for ckpt, split, domain, bucket in ordered_rows(grouped, splits):
            writer.writerow([ckpt, split, domain, bucket.n, bucket.mean_reward])

    with (root / "eval_rule_based_reward_by_ckpt_split_domain.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ckpt", "split", "domain", "n", "fallback_count", "mean_rule_based_reward"])
        for ckpt, split, domain, bucket in ordered_rows(grouped, splits):
            writer.writerow([ckpt, split, domain, bucket.n, bucket.fallback_count, bucket.mean_rule_reward])

    with (root / "eval_judge_weighted_score_by_ckpt_split_domain.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ckpt", "split", "domain", "n", "mean_judge_weighted_score"])
        for ckpt, split, domain, bucket in ordered_rows(grouped, splits):
            writer.writerow([ckpt, split, domain, bucket.n, bucket.mean_weighted])

    with (root / "eval_judge_scores_by_ckpt_split_domain.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ckpt", "split", "domain", "n", *[f"mean_{name}" for name in SCORE_NAMES]])
        for ckpt, split, domain, bucket in ordered_rows(grouped, splits):
            writer.writerow([ckpt, split, domain, bucket.n, *[bucket.mean_score(name) for name in SCORE_NAMES]])


def similarity_table_row(
    buckets: dict[tuple[str, str], Bucket], split: str, domains: tuple[str, ...]
) -> tuple[str, float]:
    total_sum = 0.0
    total_count = 0
    values: list[str] = []
    for domain in domains:
        bucket = buckets.get((split, domain))
        if bucket is not None:
            total_sum += bucket.similarity_sum
            total_count += bucket.similarity_count
            values.append(f"{bucket.mean_similarity:.6f}")
        else:
            values.append("0.000000")
    avg = total_sum / total_count if total_count else 0.0
    return " | ".join(values), avg


def write_similarity_stats(
    root: Path, grouped: dict[int, dict[tuple[str, str], Bucket]], splits: tuple[str, ...] = ("ID", "OOD")
) -> None:
    single_split = len(splits) == 1
    csv_filename = "eval_similarity_by_ckpt_domain.csv" if single_split else "eval_similarity_by_ckpt_split_domain.csv"

    with (root / csv_filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if single_split:
            writer.writerow(["ckpt", "domain", "n", "mean_similarity"])
            for ckpt, _split, domain, bucket in ordered_rows(grouped, splits):
                writer.writerow([ckpt, domain, bucket.similarity_count, bucket.mean_similarity])
        else:
            writer.writerow(["ckpt", "split", "domain", "n", "mean_similarity"])
            for ckpt, split, domain, bucket in ordered_rows(grouped, splits):
                writer.writerow([ckpt, split, domain, bucket.similarity_count, bucket.mean_similarity])


def format_domain_line(buckets: dict[tuple[str, str], Bucket], split: str, domains: tuple[str, ...], metric: str) -> str:
    parts = []
    for domain in domains:
        bucket = buckets.get((split, domain))
        if bucket is None:
            continue
        value = bucket.mean_reward if metric == "reward" else bucket.mean_weighted
        parts.append(f"{domain}={value:.6f} (n={bucket.n})")
    return ", ".join(parts)


def write_detail_summary(
    root: Path,
    grouped: dict[int, dict[tuple[str, str], Bucket]],
    metric: str,
    splits: tuple[str, ...] = ("ID", "OOD"),
) -> None:
    is_reward = metric == "reward"
    filename_suffix = "idood" if len(splits) == 2 else "domain"
    filename = f"eval_reward_by_ckpt_{filename_suffix}_summary.md" if is_reward else f"eval_judge_weighted_score_by_ckpt_{filename_suffix}_summary.md"
    metric_label = "reward" if is_reward else "five-point judge score"
    split_label = "ID-OOD" if len(splits) == 2 else splits[0]
    title = f"Eval {"reward" if is_reward else "judge weighted score"} by ckpt / {split_label} summary"
    specs = [(split, domains) for split, domains in SPLIT_SPECS if split in splits]

    lines = [
        f"# {title}",
        "",
        f"This is the {metric_label} summary. {split_label} averages are weighted by `n` across domains.",
        "",
        "## Overview",
        "",
        "| ckpt | " + " | ".join(f"{split} avg" for split, _domains in specs) + " |",
        "| --- | " + " | ".join("---:" for _ in specs) + " |",
    ]
    for ckpt in sorted(grouped):
        values = []
        for split, _domains in specs:
            bucket = aggregate_split(grouped[ckpt], split)
            value = bucket.mean_reward if is_reward else bucket.mean_weighted
            values.append(f"{value:.6f}")
        lines.append(f"| `{ckpt}` | " + " | ".join(values) + " |")

    for ckpt in sorted(grouped):
        lines.extend(["", f"## ckpt `{ckpt}`", ""])
        for split, domains in specs:
            bucket = aggregate_split(grouped[ckpt], split)
            value = bucket.mean_reward if is_reward else bucket.mean_weighted
            lines.append(f"- {split} avg: `{value:.6f}` over `n={bucket.n}`")
            lines.append(f"- {split} domains: {format_domain_line(grouped[ckpt], split, domains, metric)}")
    (root / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rule_based_summary(
    root: Path, grouped: dict[int, dict[tuple[str, str], Bucket]], splits: tuple[str, ...] = ("ID", "OOD")
) -> None:
    lines = [
        "# Rule-based judge fallback summary",
        "",
        "This report counts samples where `judge_rule_based_reward` is present or `judge_fallback` is true.",
    ]
    for split, _domains in SPLIT_SPECS:
        if split not in splits:
            continue
        lines.extend([
            "",
            f"## {split}",
            "",
            f"| ckpt | fallback_count | {split} avg rule reward |",
            "| --- | ---: | ---: |",
        ])
        for ckpt in sorted(grouped):
            bucket = aggregate_split(grouped[ckpt], split)
            lines.append(f"| `{ckpt}` | {bucket.fallback_count} | {bucket.mean_rule_reward:.6f} |")

    latest_ckpt = max(grouped)
    lines.extend([
        "",
        "## Highlights",
        "",
        f"- Latest ckpt: `{latest_ckpt}`.",
        f"- Any fallback records: `{sum(bucket.fallback_count for buckets in grouped.values() for bucket in buckets.values())}`.",
    ])
    (root / "eval_rule_based_reward_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_row(ckpt: int, buckets: dict[tuple[str, str], Bucket], split: str, domains: tuple[str, ...], *, percent: bool) -> str:
    total = aggregate_split(buckets, split)
    values = []
    for domain in domains:
        bucket = buckets.get((split, domain))
        value = bucket.mean_weighted if bucket else 0.0
        if percent:
            value *= 20.0
        values.append(f"{value:.2f}" if percent else f"{value:.6f}")
    avg = total.mean_weighted * (20.0 if percent else 1.0)
    values.append(f"{avg:.2f}" if percent else f"{avg:.6f}")
    return f"| `{ckpt}` | " + " | ".join(values) + " |"


def write_readme_like(
    root: Path,
    grouped: dict[int, dict[tuple[str, str], Bucket]],
    *,
    percent: bool,
    splits: tuple[str, ...] = ("ID", "OOD"),
) -> None:
    filename_suffix = "idood" if len(splits) == 2 else "domain"
    filename = f"eval_judge_weighted_score_by_ckpt_{filename_suffix}_summary_percent.md" if percent else "README.md"
    title = "Five-point judge weighted score summary in percent" if percent else "Five-point judge weighted score summary"
    lines = [f"# {title}", ""]
    if percent:
        lines.append("Percent values are computed as `five-point score × 20`.")
    else:
        if len(splits) == 2:
            lines.append("This README shows the five-point judge weighted score with `ID` and `OOD` split into two separate tables.")
        else:
            lines.append(f"This README shows the five-point judge weighted score with the `{splits[0]}` split.")
    lines.append("Each split average is weighted by `n` across domains.")

    for split, domains in SPLIT_SPECS:
        if split not in splits:
            continue
        lines.extend(["", f"## {split}", ""])
        lines.append("| ckpt | " + " | ".join(domains) + f" | {split} avg |")
        lines.append("| --- | " + " | ".join("---:" for _ in domains) + " | ---: |")
        for ckpt in sorted(grouped):
            lines.append(table_row(ckpt, grouped[ckpt], split, domains, percent=percent))

    lines.extend(["", "## Similarity", ""])
    lines.append("This table averages `similarity_score` across non-null samples within each domain.")
    for split, domains in SPLIT_SPECS:
        if split not in splits:
            continue
        lines.extend(["", f"### {split}", ""])
        lines.append("| ckpt | " + " | ".join(domains) + " | avg |")
        lines.append("| --- | " + " | ".join("---:" for _ in domains) + " | ---: |")
        for ckpt in sorted(grouped):
            values, avg = similarity_table_row(grouped[ckpt], split, domains)
            lines.append(f"| `{ckpt}` | {values} | {avg:.6f} |")

    latest_ckpt = max(grouped)
    scale = 20.0 if percent else 1.0
    precision = 2 if percent else 6
    lines.extend(["", "## Highlights", "", f"- Latest ckpt: `{latest_ckpt}`."])
    for split, _domains in SPLIT_SPECS:
        if split not in splits:
            continue
        best = max(sorted(grouped), key=lambda ckpt: aggregate_split(grouped[ckpt], split).mean_weighted)
        best_value = aggregate_split(grouped[best], split).mean_weighted * scale
        lines.append(f"- Best {split} avg: ckpt `{best}` with `{best_value:.{precision}f}`.")

    lines.extend(["", "Notes:"])
    if len(splits) == 2:
        lines.extend([
            "- `swe` has `n=100` for `ID` and `n=50` for `OOD`.",
            "- `web` has `n=76` for `ID` and `n=50` for `OOD`.",
            "- The remaining domains use `n=50` each.",
        ])
    elif splits == ("OOD",):
        lines.append("- Only the `OOD` split is reported; `ID` records are ignored.")
    else:
        lines.append("- Only the `ID` split is reported; `OOD` records are ignored.")
    (root / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing eval_test_*.jsonl files")
    parser.add_argument(
        "--split",
        choices=("both", "ID", "OOD"),
        default="both",
        help="Which split(s) to report: both, ID, or OOD (default: both).",
    )
    args = parser.parse_args()

    splits = ("ID", "OOD") if args.split == "both" else (args.split,)
    grouped = read_eval_files(args.root)
    if not grouped:
        raise SystemExit(f"no eval_test_*.jsonl files found under {args.root}")

    write_csvs(args.root, grouped, splits)
    write_detail_summary(args.root, grouped, "reward", splits)
    write_detail_summary(args.root, grouped, "weighted", splits)
    write_rule_based_summary(args.root, grouped, splits)
    write_readme_like(args.root, grouped, percent=False, splits=splits)
    write_readme_like(args.root, grouped, percent=True, splits=splits)
    write_similarity_stats(args.root, grouped, splits)
    print(f"summarized {len(grouped)} checkpoints through ckpt {max(grouped)} using split(s): {', '.join(splits)}")


if __name__ == "__main__":
    main()
