from __future__ import annotations

from pathlib import Path
from typing import Any


def render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "# MindMate Character Memory Eval",
        "",
        f"- user_id: `{result['user_id']}`",
        f"- cases: {result['case_count']}",
        f"- memory cases: {result['memory_case_count']}",
        "",
        "## Baseline Summary",
        "",
        "| baseline | DRA | MCR | leaked/private | prompt tokens | memory count |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["summary"].items():
        lines.append(
            "| {name} | {dra:.2f} | {mcr:.2f} | {leaked}/{private} | {tokens:.0f} | {count:.1f} |".format(
                name=name,
                dra=row["dra"],
                mcr=row["mcr"],
                leaked=row["leaked_private"],
                private=row["private_total"],
                tokens=row["prompt_tokens"],
                count=row["memory_count"],
            )
        )

    switch = result["switch_summary"]
    lines.extend([
        "",
        "## Switch Summary",
        "",
        f"- accuracy: {switch['accuracy']:.2f} ({switch['correct']}/{switch['total']})",
        "",
        "## Case Details",
        "",
    ])
    for case in result["cases"]:
        lines.append(f"### {case['case_id']} ({case['active_character_id']})")
        lines.append(f"- query: {case['query']}")
        lines.append(
            "- switch: expected `{}` / predicted `{}` / score {}".format(
                case["switch"]["expected_switch_to"],
                case["switch"]["predicted_switch_to"],
                case["switch"]["score"],
            )
        )
        if not case["memory_eval"]:
            lines.append("- memory eval: skipped (switch-only case)")
            lines.append("")
            continue
        for baseline, metrics in case["baselines"].items():
            lines.append(
                "- {baseline}: DRA={dra:.0f}, MCR={mcr:.2f}, tokens={tokens}, ids={ids}".format(
                    baseline=baseline,
                    dra=metrics["dra"],
                    mcr=metrics["mcr"],
                    tokens=metrics["prompt_tokens"],
                    ids=", ".join(metrics["external_ids"][:8]) or "-",
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result), encoding="utf-8")
