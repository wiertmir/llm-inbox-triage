#!/usr/bin/env python3
"""Evaluate batch triage against labeled categories and extracted fields."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from triage import (
    CATEGORY_STYLE,
    Category,
    CliError,
    Money,
    ModelOptions,
    add_model_arguments,
    batch_files,
    positive_interval,
    positive_jobs,
    select_provider,
    validate_model_arguments,
)

ROOT = Path(__file__).resolve().parent
FieldName = Literal["dates", "amounts", "names", "deadlines"]
FIELD_NAMES: tuple[FieldName, ...] = ("dates", "amounts", "names", "deadlines")
ExtractedValue = str | date | tuple[float, str]


class ScoredMoney(Money):
    model_config = ConfigDict(extra="forbid", strict=True)

    amount: float = Field(allow_inf_nan=False)
    currency: str = Field()


class ScoredExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dates: list[date]
    amounts: list[ScoredMoney]
    names: list[str]
    deadlines: list[date]


class ExpectedLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    category: Category
    extracted: ScoredExtraction


class ExpectedSample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    file: str = Field(min_length=1)
    expected: ExpectedLabel


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    samples: list[ExpectedSample] = Field(min_length=1)


class ScoredResult(ExpectedLabel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str = Field(min_length=1)


class ExecutionError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    error: str = Field(min_length=1)


class BatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    results: list[ScoredResult]
    errors: list[ExecutionError]


@dataclass(frozen=True)
class Dataset:
    manifest: Manifest
    directory: Path


@dataclass(frozen=True)
class SampleScore:
    sample: ExpectedSample
    actual: ScoredResult | None
    error: str | None
    category_match: bool
    field_matches: dict[FieldName, bool]

    @property
    def exact_match(self) -> bool:
        return self.category_match and all(self.field_matches.values())


@dataclass(frozen=True)
class AccuracyMetric:
    name: str
    correct: int
    total: int

    @property
    def percentage(self) -> float:
        return 100 * self.correct / self.total if self.total else 0.0


class EvalOptions(ModelOptions):
    expected: Path
    results: Path | None
    jobs: int
    request_interval: float


def unique_values(values: Iterable[str], description: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise CliError(f"Duplicate {description}: {', '.join(duplicates)}")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CliError(f"Cannot read {path}: {exc}") from exc


def load_dataset(path: Path) -> Dataset:
    try:
        manifest = Manifest.model_validate_json(read_text(path))
    except ValidationError as exc:
        raise CliError(f"Invalid expected-results manifest {path}: {exc}") from exc
    unique_values((sample.id for sample in manifest.samples), "sample IDs")
    root = path.resolve().parent
    files = []
    for sample in manifest.samples:
        relative = Path(sample.file)
        if relative.is_absolute() or ".." in relative.parts:
            raise CliError(f"Sample path must stay within the manifest directory: {sample.file}")
        try:
            source = (root / relative).resolve()
        except (OSError, ValueError) as exc:
            raise CliError(f"Invalid sample path {sample.file!r}: {exc}") from exc
        if not source.is_relative_to(root) or source.suffix.lower() != ".txt":
            raise CliError(f"Invalid sample text-file path: {sample.file}")
        if not source.is_file():
            raise CliError(f"Sample file not found: {source}")
        files.append(source)
    unique_values((source.name for source in files), "input filenames")
    directory = files[0].parent
    if any(source.parent != directory for source in files):
        raise CliError("All labeled samples must be directly in one batch directory")
    discovered = set(batch_files(str(directory)))
    if discovered != set(files):
        unlabeled = sorted(source.name for source in discovered - set(files))
        raise CliError(f"Batch directory contains unlabeled .txt files: {', '.join(unlabeled)}")
    return Dataset(manifest, directory)


def parse_batch_output(text: str) -> BatchPayload:
    try:
        payload = BatchPayload.model_validate_json(text)
    except ValidationError as exc:
        raise CliError(f"Invalid batch JSON output: {exc}") from exc
    unique_values(
        [result.id for result in payload.results] + [error.id for error in payload.errors],
        "batch result/error IDs",
    )
    return payload


def run_batch_cli(dataset: Dataset, args: EvalOptions) -> tuple[BatchPayload, int]:
    command = [
        sys.executable, str(ROOT / "triage.py"),
        f"--batch={dataset.directory}", "--json",
    ]
    if args.ollama is not None:
        command.append(f"--ollama={args.ollama}")
    else:
        command.extend(["--provider", args.provider])
    command.extend([
        "--jobs", str(args.jobs), "--request-interval", str(args.request_interval),
    ])
    if args.model is not None:
        command.extend(["--model", args.model])
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", check=False,
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise CliError(f"Cannot run batch triage: {exc}") from exc
    if not completed.stdout.strip():
        raise CliError(f"Batch triage exited with code {completed.returncode} and produced no JSON")
    return parse_batch_output(completed.stdout), completed.returncode


def normalized_fields(extracted: ScoredExtraction) -> dict[FieldName, frozenset[ExtractedValue]]:
    return {
        "dates": frozenset(extracted.dates),
        "amounts": frozenset((money.amount, money.currency) for money in extracted.amounts),
        "names": frozenset(extracted.names),
        "deadlines": frozenset(extracted.deadlines),
    }


def score_samples(manifest: Manifest, payload: BatchPayload) -> list[SampleScore]:
    expected_ids = {Path(sample.file).name for sample in manifest.samples}
    received_ids = {result.id for result in payload.results} | {error.id for error in payload.errors}
    unexpected = sorted(received_ids - expected_ids)
    if unexpected:
        raise CliError(f"Batch returned unlabeled input IDs: {', '.join(unexpected)}")
    results = {result.id: result for result in payload.results}
    errors = {error.id: error.error for error in payload.errors}
    scores = []
    for sample in manifest.samples:
        filename = Path(sample.file).name
        actual = results.get(filename)
        if actual is None:
            scores.append(SampleScore(
                sample, None, errors.get(filename, "No result returned by batch triage"),
                False, {field: False for field in FIELD_NAMES},
            ))
            continue
        expected_fields = normalized_fields(sample.expected.extracted)
        actual_fields = normalized_fields(actual.extracted)
        scores.append(SampleScore(
            sample, actual, None, actual.category == sample.expected.category,
            {field: expected_fields[field] == actual_fields[field] for field in FIELD_NAMES},
        ))
    return scores


def accuracy_metrics(scores: list[SampleScore]) -> list[AccuracyMetric]:
    total = len(scores)
    return [
        AccuracyMetric("Category", sum(score.category_match for score in scores), total),
        *(AccuracyMetric(
            field.capitalize(), sum(score.field_matches[field] for score in scores), total,
        ) for field in FIELD_NAMES),
        AccuracyMetric(
            "Key fields (4 checks/sample)",
            sum(sum(score.field_matches.values()) for score in scores), total * len(FIELD_NAMES),
        ),
        AccuracyMetric("Exact sample (category + fields)", sum(score.exact_match for score in scores), total),
    ]


def percentage_text(correct: int, total: int) -> Text:
    if not total:
        return Text("N/A", style="dim")
    style = "bold green" if correct == total else "bold yellow" if correct else "bold red"
    return Text(f"{100 * correct / total:.1f}%", style=style)


def display_values(values: frozenset[ExtractedValue]) -> str:
    formatted = [
        value.isoformat() if isinstance(value, date)
        else f"{value[0]} {value[1]}" if isinstance(value, tuple)
        else value
        for value in sorted(values, key=str)
    ]
    return json.dumps(formatted, ensure_ascii=True)


def render_report(
    scores: list[SampleScore], console: Console, source: str, elapsed: float, exit_code: int | None,
) -> None:
    successful = sum(score.actual is not None for score in scores)
    failed = len(scores) - successful
    console.print(Panel(
        Text(
            f"{source}\n"
            f"{len(scores)} labeled samples | {successful} returned | {failed} failed/missing | "
            f"{elapsed:.1f}s\n"
            f"Batch exit code: {exit_code if exit_code is not None else 'N/A (saved JSON)'}"
        ),
        title="Inbox triage | Accuracy report", border_style="cyan", box=box.ROUNDED,
    ))
    metrics = Table(
        title="Accuracy metrics", title_justify="left",
        box=box.ROUNDED, border_style="cyan", header_style="bold cyan",
        caption="Every labeled sample is scored. Failed/missing results are incorrect on every check.",
        caption_justify="left",
    )
    metrics.add_column("Metric", ratio=2)
    metrics.add_column("Correct", justify="right")
    metrics.add_column("Total", justify="right")
    metrics.add_column("Accuracy", justify="right")
    for metric in accuracy_metrics(scores):
        metrics.add_row(
            metric.name, str(metric.correct), str(metric.total),
            percentage_text(metric.correct, metric.total),
        )
    console.print(metrics)

    categories = Table(
        title="Category accuracy", title_justify="left",
        box=box.ROUNDED, border_style="cyan", header_style="bold cyan",
    )
    categories.add_column("Expected category")
    categories.add_column("Samples", justify="right")
    categories.add_column("Correct", justify="right")
    categories.add_column("Accuracy", justify="right")
    for category in Category:
        matching = [score for score in scores if score.sample.expected.category == category]
        correct = sum(score.category_match for score in matching)
        categories.add_row(
            Text(category.value, style=CATEGORY_STYLE[category][1]),
            str(len(matching)), str(correct), percentage_text(correct, len(matching)),
        )
    console.print(categories)

    differences = Table(
        title="Mismatches and execution failures", title_justify="left",
        box=box.ROUNDED, border_style="yellow", header_style="bold yellow",
    )
    differences.add_column("File", overflow="fold")
    differences.add_column("Check")
    differences.add_column("Expected", overflow="fold")
    differences.add_column("Actual", overflow="fold")
    for score in scores:
        filename = json.dumps(Path(score.sample.file).name, ensure_ascii=True)[1:-1]
        if score.actual is None:
            differences.add_row(
                Text(filename), "Execution", "A result",
                Text(json.dumps(score.error, ensure_ascii=True)), style="red",
            )
            continue
        if not score.category_match:
            differences.add_row(
                Text(filename), "Category", score.sample.expected.category.value,
                score.actual.category.value,
            )
        expected_fields = normalized_fields(score.sample.expected.extracted)
        actual_fields = normalized_fields(score.actual.extracted)
        for field in FIELD_NAMES:
            if not score.field_matches[field]:
                differences.add_row(
                    Text(filename), field.capitalize(), Text(display_values(expected_fields[field])),
                    Text(display_values(actual_fields[field])),
                )
    if differences.row_count:
        console.print(differences)
    else:
        console.print(Text("All labeled samples matched category and extracted fields.", style="bold green"))
    console.print(Text(
        "Lists are unordered sets; extra values fail a check. Names are case-sensitive.\n"
        "Amounts match both numeric value and currency. Priority, summary, replies and events are unscored.",
        style="dim",
    ))
    if exit_code or failed:
        console.print(Text("Execution incomplete or failed; scores include these failures.", style="bold red"))


def parse_arguments() -> EvalOptions:
    parser = argparse.ArgumentParser(description="Evaluate inbox triage against labeled samples")
    parser.add_argument(
        "--expected", type=Path, default=ROOT / "evals" / "expected.json",
        help="Labeled manifest (default: evals/expected.json)",
    )
    add_model_arguments(parser)
    parser.add_argument("--jobs", type=positive_jobs, default=3, help="Concurrent batch workers (default: 3)")
    parser.add_argument(
        "--request-interval", type=positive_interval, default=1.0, metavar="SECONDS",
        help="Minimum batch AI request spacing, including retries (default: 1)",
    )
    parser.add_argument(
        "--results", type=Path,
        help="Evaluate an existing batch JSON instead of making new API requests",
    )
    args = EvalOptions()
    parser.parse_args(namespace=args)
    validate_model_arguments(parser, args)
    return args


def main() -> int:
    args = parse_arguments()
    console = Console()
    started = time.perf_counter()
    try:
        dataset = load_dataset(args.expected)
        if args.results is not None:
            payload = parse_batch_output(read_text(args.results))
            exit_code = None
            source = "Saved batch JSON (provider not recorded)"
        else:
            payload, exit_code = run_batch_cli(dataset, args)
            selected = select_provider(args.provider, args.model, args.ollama)
            source = (
                f"Provider: {selected.name} | Model: {json.dumps(selected.model, ensure_ascii=True)}\n"
                f"Jobs: {args.jobs} | Request spacing: {args.request_interval:g}s"
            )
            if args.ollama is not None:
                source += f"\nOllama endpoint: {args.ollama}"
        source += (
            f"\nDataset: {json.dumps(args.expected.name, ensure_ascii=True)} | "
            f"Schema version: {dataset.manifest.schema_version}"
        )
        scores = score_samples(dataset.manifest, payload)
        render_report(scores, console, source, time.perf_counter() - started, exit_code)
        return 1 if exit_code or any(score.error is not None for score in scores) else 0
    except KeyboardInterrupt:
        Console(stderr=True).print(Text("ERROR | Evaluation interrupted.", style="bold red"))
        return 130
    except CliError as exc:
        Console(stderr=True).print(Text(f"ERROR | {exc}", style="bold red"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
