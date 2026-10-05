"""Offline tests for the issue #8 evaluator; never invoke a real AI service."""

import copy
import asyncio
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console
from rich.progress import Progress

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_triage as evaluator  # noqa: E402
import triage  # noqa: E402


def empty_fields():
    return {"dates": [], "amounts": [], "names": [], "deadlines": []}


@pytest.fixture
def manifest_path(tmp_path):
    folder = tmp_path / "evals"
    samples = folder / "samples"
    samples.mkdir(parents=True)
    (samples / "invoice.txt").write_text("Subject: An invoice\n", encoding="utf-8")
    (samples / "question.txt").write_text("Subject: A question\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "samples": [
            {
                "id": "invoice",
                "file": "samples/invoice.txt",
                "expected": {
                    "category": "invoice",
                    "extracted": {
                        "dates": ["2030-02-01", "2030-02-02"],
                        "amounts": [
                            {"amount": 100, "currency": "USD"},
                            {"amount": 120, "currency": "EUR"},
                        ],
                        "names": ["Alice Rowan", "Northstar Labs"],
                        "deadlines": ["2030-02-02"],
                    },
                },
            },
            {
                "id": "question",
                "file": "samples/question.txt",
                "expected": {"category": "question", "extracted": empty_fields()},
            },
        ],
    }
    path = folder / "expected.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def dataset(manifest_path):
    return evaluator.load_dataset(manifest_path)


@pytest.fixture
def raw_batch(dataset):
    return {
        "results": [
            {
                "id": Path(sample.file).name,
                **sample.expected.model_dump(mode="json"),
                "priority": 2,
                "summary": "Summary is not scored.",
                "suggested_reply": None,
                "proposed_events": [],
            }
            for sample in dataset.manifest.samples
        ],
        "errors": [],
    }


def score(dataset, raw):
    return evaluator.score_samples(
        dataset.manifest, evaluator.parse_batch_output(json.dumps(raw)),
    )


def test_real_manifest_loads_twenty_samples():
    dataset = evaluator.load_dataset(ROOT / "evals" / "expected.json")
    assert len(dataset.manifest.samples) == 20
    assert dataset.directory == ROOT / "evals" / "samples"


def test_exact_results_score_one_hundred_percent(dataset, raw_batch):
    scores = score(dataset, raw_batch)
    assert all(item.exact_match for item in scores)
    metrics = evaluator.accuracy_metrics(scores)
    assert len(metrics) == 7
    assert all(metric.percentage == 100 for metric in metrics)
    assert metrics[-2].correct == metrics[-2].total == 8
    assert metrics[-1].correct == metrics[-1].total == 2


def test_order_duplicates_and_numeric_representation_do_not_affect_scores(dataset, raw_batch):
    raw_batch["results"].reverse()
    fields = raw_batch["results"][1]["extracted"]
    for key in fields:
        fields[key].reverse()
        fields[key] += copy.deepcopy(fields[key])
    fields["amounts"][0]["amount"] = 120.0

    scores = score(dataset, raw_batch)

    assert all(item.exact_match for item in scores)
    assert [item.actual.id for item in scores if item.actual is not None] == ["invoice.txt", "question.txt"]


@pytest.mark.parametrize("field,value", [
    ("dates", ["2030-02-01"]),
    ("dates", ["2030-02-01", "2030-02-02", "2030-02-03"]),
    ("names", ["alice rowan", "Northstar Labs"]),
    ("names", ["Alice Rowan", "Northstar Labs", "Unexpected Person"]),
    ("deadlines", []),
    ("amounts", [{"amount": 100, "currency": "JPY"}, {"amount": 120, "currency": "EUR"}]),
    ("amounts", [{"amount": 100, "currency": "usd"}, {"amount": 120, "currency": "EUR"}]),
    ("amounts", [{"amount": 100.00000000001, "currency": "USD"}, {"amount": 120, "currency": "EUR"}]),
])
def test_field_mismatches_fail_only_the_affected_check(dataset, raw_batch, field, value):
    raw_batch["results"][0]["extracted"][field] = value

    scores = score(dataset, raw_batch)

    assert scores[0].category_match
    assert not scores[0].exact_match
    assert scores[0].field_matches[field] is False
    assert sum(scores[0].field_matches.values()) == 3
    metrics = {metric.name: metric for metric in evaluator.accuracy_metrics(scores)}
    assert metrics["Key fields (4 checks/sample)"].percentage == 87.5
    assert metrics["Exact sample (category + fields)"].percentage == 50


def test_empty_expected_fields_penalize_hallucinated_values(dataset, raw_batch):
    raw_batch["results"][1]["extracted"]["names"] = ["An invented name"]
    scores = score(dataset, raw_batch)
    assert scores[1].field_matches["names"] is False


def test_category_accuracy_is_separate_from_extraction(dataset, raw_batch):
    raw_batch["results"][0]["category"] = "urgent"
    scores = score(dataset, raw_batch)
    assert not scores[0].category_match
    assert all(scores[0].field_matches.values())
    metrics = {metric.name: metric for metric in evaluator.accuracy_metrics(scores)}
    assert metrics["Category"].percentage == 50
    assert metrics["Key fields (4 checks/sample)"].percentage == 100
    assert metrics["Exact sample (category + fields)"].percentage == 50


@pytest.mark.parametrize("explicit_error", [False, True])
def test_failed_and_missing_results_are_in_every_denominator(dataset, raw_batch, explicit_error):
    raw_batch["results"].pop(0)
    if explicit_error:
        raw_batch["errors"].append({"id": "invoice.txt", "error": "API unavailable"})
    scores = score(dataset, raw_batch)
    assert scores[0].actual is None
    assert scores[0].error
    assert not any(scores[0].field_matches.values())
    assert all(metric.percentage == 50 for metric in evaluator.accuracy_metrics(scores))
    assert evaluator.accuracy_metrics(scores)[-2].total == 8


def test_all_missing_results_score_zero(dataset):
    scores = score(dataset, {"results": [], "errors": []})
    assert len(scores) == 2
    assert all(metric.percentage == 0 for metric in evaluator.accuracy_metrics(scores))


@pytest.mark.parametrize("mode", ["duplicate-result", "result-and-error", "missing-field", "bad-currency", "nonfinite"])
def test_malformed_batch_shapes_fail_explicitly(raw_batch, mode):
    if mode == "duplicate-result":
        raw_batch["results"].append(copy.deepcopy(raw_batch["results"][0]))
    elif mode == "result-and-error":
        raw_batch["errors"].append({"id": "invoice.txt", "error": "failed"})
    elif mode == "missing-field":
        del raw_batch["results"][1]["extracted"]["names"]
    elif mode == "bad-currency":
        del raw_batch["results"][0]["extracted"]["amounts"][0]["currency"]
    elif mode == "nonfinite":
        raw_batch["results"][0]["extracted"]["amounts"][0]["amount"] = float("nan")
    with pytest.raises(evaluator.CliError):
        evaluator.parse_batch_output(json.dumps(raw_batch))


@pytest.mark.parametrize("text", ["", "not json", "[]", '{"results": []}', '{"results": [], "errors": [], "extra": 1}'])
def test_invalid_batch_json_is_not_success_shaped(text):
    with pytest.raises(evaluator.CliError, match="Invalid batch JSON"):
        evaluator.parse_batch_output(text)


def test_unexpected_batch_ids_are_rejected(dataset, raw_batch):
    raw_batch["results"][0]["id"] = "unlabeled.txt"
    with pytest.raises(evaluator.CliError, match="unlabeled input IDs"):
        score(dataset, raw_batch)


@pytest.mark.parametrize("mode", [
    "empty", "version", "duplicate-id", "duplicate-file", "bad-category",
    "missing-key", "missing-currency", "wrong-field-type", "traversal", "missing-file",
])
def test_invalid_manifests_fail_before_requests(manifest_path, mode):
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mode == "empty":
        raw["samples"] = []
    elif mode == "version":
        raw["schema_version"] = 2
    elif mode == "duplicate-id":
        raw["samples"][1]["id"] = raw["samples"][0]["id"]
    elif mode == "duplicate-file":
        raw["samples"][1]["file"] = raw["samples"][0]["file"]
    elif mode == "bad-category":
        raw["samples"][0]["expected"]["category"] = "unknown"
    elif mode == "missing-key":
        del raw["samples"][1]["expected"]["extracted"]["dates"]
    elif mode == "missing-currency":
        del raw["samples"][0]["expected"]["extracted"]["amounts"][0]["currency"]
    elif mode == "wrong-field-type":
        raw["samples"][0]["expected"]["extracted"]["names"] = "Alice"
    elif mode == "traversal":
        raw["samples"][0]["file"] = "../outside.txt"
    elif mode == "missing-file":
        raw["samples"][0]["file"] = "samples/missing.txt"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(evaluator.CliError):
        evaluator.load_dataset(manifest_path)


def test_unlabeled_text_files_are_rejected(manifest_path):
    (manifest_path.parent / "samples" / "extra.txt").write_text("Unexpected input", encoding="utf-8")
    with pytest.raises(evaluator.CliError, match="unlabeled .txt files"):
        evaluator.load_dataset(manifest_path)


def test_samples_in_different_directories_are_rejected(manifest_path):
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["samples"][1]["file"] = "question.txt"
    (manifest_path.parent / "question.txt").write_text("Question", encoding="utf-8")
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(evaluator.CliError, match="one batch directory"):
        evaluator.load_dataset(manifest_path)


def options(manifest_path, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["eval_triage.py", "--expected", str(manifest_path), *extra])
    return evaluator.parse_arguments()


def test_subprocess_command_uses_batch_json_same_interpreter_and_utf8(
    dataset, manifest_path, raw_batch, monkeypatch,
):
    seen = {}

    def run(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(raw_batch))

    monkeypatch.setattr(evaluator.subprocess, "run", run)
    args = options(
        manifest_path, monkeypatch, "--provider", "anthropic", "--jobs", "2",
        "--request-interval", "2.5",
    )

    payload, code = evaluator.run_batch_cli(dataset, args)

    assert code == 0 and len(payload.results) == 2
    assert seen["command"] == [
        sys.executable, str(ROOT / "triage.py"), f"--batch={dataset.directory}",
        "--json", "--provider", "anthropic", "--jobs", "2", "--request-interval", "2.5",
    ]
    assert seen["cwd"] == ROOT
    assert seen["stdin"] == subprocess.DEVNULL
    assert seen["stdout"] == subprocess.PIPE
    assert seen.get("stderr") is None
    assert seen["encoding"] == "utf-8"
    assert seen["env"]["PYTHONIOENCODING"] == "utf-8"
    assert seen["env"]["PYTHONUTF8"] == "1"
    assert seen["check"] is False
    assert "--create-events" not in seen["command"]


def test_evaluator_inherits_fifteen_initial_status_rows(monkeypatch, tmp_path):
    dataset = evaluator.load_dataset(ROOT / "evals" / "expected.json")
    args = options(ROOT / "evals" / "expected.json", monkeypatch)
    instances: list[Progress] = []
    visible_at_request = []
    original_progress = triage.Progress

    def progress(*args, **kwargs):
        instance = original_progress(*args, **kwargs)
        instances.append(instance)
        return instance

    async def fake_triage(text):
        visible_at_request.append(sum(task.visible for task in instances[0].tasks))
        await asyncio.sleep(0)
        return triage.TriageAnalysis(
            category=triage.Category.question, priority=2, summary="Offline test.",
            extracted=triage.Extracted(),
        )

    def run(command, **kwargs):
        output = io.StringIO()
        with monkeypatch.context() as child:
            child.setattr(sys, "argv", command[1:])
            child.setattr(sys, "stdout", output)
            code = triage.main()
        return subprocess.CompletedProcess(command, code, stdout=output.getvalue())

    monkeypatch.setattr(triage, "Progress", progress)
    monkeypatch.setattr(triage, "triage_openai", fake_triage)
    monkeypatch.setattr(triage, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(evaluator.subprocess, "run", run)

    payload, code = evaluator.run_batch_cli(dataset, args)

    assert code == 0
    assert len(payload.results) == 20
    assert visible_at_request[:3] == [15, 15, 15]
    assert max(visible_at_request) == 20
    assert all(task.visible and task.finished for task in instances[0].tasks)


def test_main_scores_valid_json_even_if_child_exits_nonzero(
    manifest_path, raw_batch, monkeypatch, capsys,
):
    options(manifest_path, monkeypatch)
    monkeypatch.setattr(
        evaluator.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, stdout=json.dumps(raw_batch)),
    )

    assert evaluator.main() == 1
    out, err = capsys.readouterr()
    assert "100.0%" in out
    assert "Batch exit code: 1" in out
    assert "Execution incomplete" in out
    assert err == ""


def test_saved_results_render_report_without_subprocess(manifest_path, raw_batch, monkeypatch, capsys):
    path = manifest_path.parent / "batch.json"
    path.write_text(json.dumps(raw_batch), encoding="utf-8")
    options(manifest_path, monkeypatch, "--results", str(path))

    def unexpected(*args, **kwargs):
        raise AssertionError("Saved-results evaluation must not invoke triage")

    monkeypatch.setattr(evaluator.subprocess, "run", unexpected)

    assert evaluator.main() == 0
    out, err = capsys.readouterr()
    assert "Accuracy report" in out
    assert "Accuracy metrics" in out
    assert "Category accuracy" in out
    assert "100.0%" in out
    assert "provider not recorded" in out
    assert "Batch exit code: N/A (saved JSON)" in out
    assert "All labeled samples matched" in out
    assert "N/A" in out
    assert err == ""


def test_accuracy_mismatches_do_not_fail_the_command(manifest_path, raw_batch, monkeypatch, capsys):
    raw_batch["results"][0]["category"] = "urgent"
    options(manifest_path, monkeypatch)
    monkeypatch.setattr(
        evaluator.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=json.dumps(raw_batch)),
    )

    assert evaluator.main() == 0
    out, _ = capsys.readouterr()
    assert "50.0%" in out
    assert "Mismatches and execution failures" in out
    assert "invoice.txt" in out
    assert "urgent" in out


def test_saved_results_with_missing_sample_exit_nonzero(manifest_path, raw_batch, monkeypatch, capsys):
    raw_batch["results"].pop(0)
    path = manifest_path.parent / "partial.json"
    path.write_text(json.dumps(raw_batch), encoding="utf-8")
    options(manifest_path, monkeypatch, "--results", str(path))

    assert evaluator.main() == 1
    out, _ = capsys.readouterr()
    assert "50.0%" in out
    assert "No result returned by batch triage" in out


@pytest.mark.parametrize("mode", ["no-json", "invalid-json", "launch-failure", "missing-manifest"])
def test_cli_errors_are_explicit_without_tracebacks(manifest_path, monkeypatch, capsys, mode):
    options(manifest_path, monkeypatch)
    if mode == "missing-manifest":
        manifest_path.unlink()

    def run(command, **kwargs):
        if mode == "launch-failure":
            raise OSError("Cannot start process")
        return subprocess.CompletedProcess(command, 1, stdout="" if mode == "no-json" else "bad json")

    monkeypatch.setattr(evaluator.subprocess, "run", run)

    assert evaluator.main() == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "ERROR" in err
    assert "Traceback" not in err


def test_ctrl_c_is_reported_cleanly(manifest_path, monkeypatch, capsys):
    options(manifest_path, monkeypatch)

    def run(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(evaluator.subprocess, "run", run)
    assert evaluator.main() == 130
    assert "interrupted" in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [("--jobs", "0"), ("--request-interval", "nan")])
def test_invalid_throttling_is_a_usage_error(manifest_path, monkeypatch, option, value):
    with pytest.raises(SystemExit) as excinfo:
        options(manifest_path, monkeypatch, option, value)
    assert excinfo.value.code == 2


def test_report_shows_exact_mismatch_values_and_preserves_numeric_precision(dataset, raw_batch):
    raw_batch["results"][0]["extracted"]["amounts"][0]["amount"] = 100.00000000001
    raw_batch["results"][1]["extracted"]["names"] = ["[bold]not markup[/bold]"]
    scores = score(dataset, raw_batch)
    output = io.StringIO()

    evaluator.render_report(
        scores, Console(file=output, width=140, color_system=None), "Offline test", 1.25, 0,
    )

    report = output.getvalue()
    assert "100.00000000001 USD" in report
    assert "[bold]not markup[/bold]" in report
    assert "100.0 USD" in report
    assert "Amounts" in report
    assert "Names" in report
    assert "case-sensitive" in report


def test_report_renders_on_legacy_windows_console(dataset, raw_batch):
    scores = score(dataset, raw_batch)
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="cp1252")

    evaluator.render_report(
        scores, Console(file=output, width=80, color_system=None, force_terminal=False),
        "Offline test", 0, 0,
    )
    output.flush()

    report = buffer.getvalue().decode("cp1252")
    assert "100.0%" in report
    assert "Accuracy report" in report


def test_unicode_manifest_name_renders_on_legacy_console(manifest_path, raw_batch, monkeypatch):
    renamed = manifest_path.with_name("expected-\u65e5\u672c.json")
    manifest_path.rename(renamed)
    results = renamed.parent / "batch.json"
    results.write_text(json.dumps(raw_batch), encoding="utf-8")
    options(renamed, monkeypatch, "--results", str(results))
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", output)

    assert evaluator.main() == 0
    output.flush()

    rendered = buffer.getvalue().decode("cp1252")
    assert "\\u65e5\\u672c" in rendered
    assert "100.0%" in rendered
