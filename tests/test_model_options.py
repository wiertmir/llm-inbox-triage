"""Shared CLI model/backend selection and Ollama HTTP contract tests."""

import asyncio
import io
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_triage as evaluator  # noqa: E402
import triage  # noqa: E402

if TYPE_CHECKING:
    import httpx2 as httpx
else:
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx


def parse_options(script, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", [f"{script}.py", *args])
    if script == "triage":
        return triage.parse_arguments()[1]
    return evaluator.parse_arguments()


@pytest.mark.parametrize("script", ["triage", "eval_triage"])
def test_default_provider_models_are_unchanged(script, monkeypatch):
    args = parse_options(script, monkeypatch)
    assert args.provider == "openai"
    assert args.model is None
    assert args.ollama is None
    assert triage.select_provider(args.provider).model == triage.OPENAI_MODEL
    assert triage.select_provider("anthropic").model == triage.ANTHROPIC_MODEL


@pytest.mark.parametrize("script", ["triage", "eval_triage"])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_cloud_model_flag_is_shared(script, provider, monkeypatch):
    args = parse_options(script, monkeypatch, "--provider", provider, "--model", "custom-model")
    selected = triage.select_provider(args.provider, args.model, args.ollama)
    assert selected.model == "custom-model"
    assert args.ollama is None


@pytest.mark.parametrize("script", ["triage", "eval_triage"])
@pytest.mark.parametrize("args", [
    ["--provider", "openai", "--ollama=pop-os.local", "--model", "qwen3.6:35b"],
    ["--ollama=pop-os.local", "--provider", "anthropic", "--model", "qwen3.6:35b"],
    ["--provider=openai", "--ollama", "127.0.0.1", "--model", "qwen3.6:35b"],
])
def test_provider_and_ollama_are_mutually_exclusive(script, args, monkeypatch, capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_options(script, monkeypatch, *args)
    assert excinfo.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("script", ["triage", "eval_triage"])
def test_ollama_requires_explicit_model(script, monkeypatch, capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_options(script, monkeypatch, "--ollama=pop-os.local")
    assert excinfo.value.code == 2
    assert "--ollama requires --model" in capsys.readouterr().err


@pytest.mark.parametrize("script", ["triage", "eval_triage"])
def test_empty_model_is_rejected(script, monkeypatch, capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_options(script, monkeypatch, "--model", " \t ")
    assert excinfo.value.code == 2
    assert "model must not be empty" in capsys.readouterr().err


@pytest.mark.parametrize("host,expected", [
    ("pop-os.local", "http://pop-os.local:11434/v1"),
    ("127.0.0.1", "http://127.0.0.1:11434/v1"),
    ("localhost:11435", "http://localhost:11435/v1"),
    ("http://pop-os.local:11434/v1/", "http://pop-os.local:11434/v1"),
    ("https://models.local:8443", "https://models.local:8443/v1"),
    ("::1", "http://[::1]:11434/v1"),
    ("[2001:db8::1]:11435", "http://[2001:db8::1]:11435/v1"),
    (" localhost ", "http://localhost:11434/v1"),
])
def test_ollama_host_normalization(host, expected):
    assert triage.ollama_base_url(host) == expected


@pytest.mark.parametrize("script", ["triage", "eval_triage"])
@pytest.mark.parametrize("host", [
    "", " ", "ftp://localhost", "http://", "http://user:password@localhost",
    "http://localhost/v1?key=test", "http://localhost/v1#fragment",
    "http://localhost/api", "localhost:invalid", "localhost:0",
    "localhost:65536", "host name", "localhost\\invalid",
    "localhost:", "local\nhost",
])
def test_invalid_ollama_hosts_are_usage_errors(script, host, monkeypatch, capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_options(script, monkeypatch, "--ollama", host, "--model", "qwen-test")
    assert excinfo.value.code == 2
    assert "--ollama" in capsys.readouterr().err


def analysis():
    return triage.TriageAnalysis(
        category=triage.Category.question, priority=2, summary="A question.",
        extracted=triage.Extracted(),
    )


def chat_response(content=None, *, refusal=None, finish="stop", empty_choices=False):
    if content is None and refusal is None:
        content = analysis().model_dump_json()
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "qwen-test",
        "choices": [] if empty_choices else [{
            "index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": content, "refusal": refusal},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def server(monkeypatch):
    requests = []
    clients = []
    responses = [(200, chat_response())]
    original_client = triage.AsyncOpenAI

    async def handler(request):
        requests.append(request)
        index = min(len(requests) - 1, len(responses) - 1)
        status, payload = responses[index]
        return httpx.Response(status, json=payload)

    def create(**kwargs):
        client = original_client(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(triage, "AsyncOpenAI", create)
    monkeypatch.setattr(triage, "_wait", lambda state: 0)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return requests, clients, responses


def test_ollama_uses_real_sdk_chat_schema_and_requested_model(server):
    requests, clients, _ = server
    selected = triage.select_provider(
        "openai", model="qwen3.6:35b", ollama=triage.ollama_base_url("pop-os.local"),
    )

    result = asyncio.run(selected.triage("Message to triage"))

    assert result == analysis()
    assert selected.name == "Ollama"
    assert selected.model == "qwen3.6:35b"
    assert selected.key_env is None
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "http://pop-os.local:11434/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer ollama"
    data = json.loads(request.content)
    assert data["model"] == "qwen3.6:35b"
    assert data["messages"] == [
        {"role": "system", "content": triage.SYSTEM_PROMPT},
        {"role": "user", "content": "Message to triage"},
    ]
    assert data["temperature"] == 0
    assert data["stream"] is False
    schema = data["response_format"]
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["strict"] is True
    assert set(schema["json_schema"]["schema"]["properties"]) == set(triage.TriageAnalysis.model_fields)
    assert "id" not in schema["json_schema"]["schema"]["properties"]
    assert all(client.is_closed() for client in clients)


def test_ollama_retry_reuses_existing_policy_and_closes_client(server):
    requests, clients, responses = server
    responses[:] = [
        (429, {"error": {"message": "Busy", "type": "rate_limit_error"}}),
        (200, chat_response()),
    ]

    result = asyncio.run(triage.triage_ollama("Message", "qwen-test", "http://localhost:11434/v1"))

    assert result == analysis()
    assert len(requests) == 2
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize("payload,description", [
    (chat_response(refusal="Cannot help"), "refused"),
    (chat_response(finish="length"), "cut off"),
    (chat_response(finish="content_filter"), "content filter"),
    (chat_response(finish="tool_calls"), "did not complete"),
    (chat_response(empty_choices=True), "no completion choices"),
    (chat_response(content=""), "no structured"),
])
def test_ollama_refusal_and_incomplete_results_are_explicit(server, payload, description):
    requests, clients, responses = server
    responses[:] = [(200, payload)]

    with pytest.raises(triage.TriageRefusal, match=description):
        asyncio.run(triage.triage_ollama("Message", "qwen-test", "http://localhost:11434/v1"))

    assert len(requests) == 1
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize("mode", ["malformed", "authentication", "missing-model"])
def test_ollama_cli_errors_are_readable_and_do_not_fall_back(server, monkeypatch, capsys, mode):
    requests, clients, responses = server
    if mode == "malformed":
        responses[:] = [(200, chat_response(content="not JSON"))]
    elif mode == "authentication":
        responses[:] = [(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})]
    else:
        responses[:] = [(404, {"error": {"message": "model not found", "type": "not_found_error"}})]
    monkeypatch.setattr(
        sys, "argv", ["triage.py", "--ollama=localhost", "--model", "qwen-test", "--json"],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("Message"))

    assert triage.main() == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert "Ollama" in err
    assert "Traceback" not in err
    assert "OPENAI_API_KEY" not in err
    assert len(requests) == 1
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize("batch_mode", [False, True])
def test_ollama_file_and_batch_modes_assign_ids_and_preserve_aggregate_shape(
    server, monkeypatch, tmp_path, capsys, batch_mode,
):
    requests, clients, _ = server
    folder = tmp_path / "messages"
    folder.mkdir()
    path = folder / "question.txt"
    path.write_text("A question", encoding="utf-8")
    monkeypatch.setattr(triage, "OUT_DIR", tmp_path / "out")
    argv = ["triage.py", "--ollama=pop-os.local", "--model", "qwen3.6:35b", "--json"]
    argv += ["--batch", str(folder), "--request-interval", "0.001"] if batch_mode else [str(path)]
    monkeypatch.setattr(sys, "argv", argv)

    assert triage.main() == 0

    data = json.loads(capsys.readouterr().out)
    if batch_mode:
        assert set(data) == {"results", "errors"}
        assert data["errors"] == []
        assert data["results"][0]["id"] == "question.txt"
    else:
        assert data["id"] == "question.txt"
    assert len(requests) == 1
    assert all(client.is_closed() for client in clients)


def test_ollama_stdin_id_is_application_owned(server, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["triage.py", "--ollama=localhost", "--model", "qwen-test", "--json"],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("Question"))

    assert triage.main() == 0

    assert json.loads(capsys.readouterr().out)["id"] == "***stdin***"


def test_custom_model_markup_does_not_break_console_status(monkeypatch):
    async def fake(text, model=None):
        return analysis()

    monkeypatch.setattr(triage, "triage_openai", fake)
    provider = triage.select_provider("openai", model="[custom]model[/custom]")
    console = Console(file=io.StringIO())

    assert triage.run_triage("Question", provider, console) == analysis()


@pytest.mark.parametrize("backend", ["openai", "anthropic", "ollama"])
def test_evaluator_forwards_requested_backend_and_model(monkeypatch, backend):
    argv = ["eval_triage.py", "--model", "custom-model"]
    argv += ["--ollama=pop-os.local"] if backend == "ollama" else ["--provider", backend]
    monkeypatch.setattr(sys, "argv", argv)
    args = evaluator.parse_arguments()
    dataset = evaluator.load_dataset(ROOT / "evals" / "expected.json")
    seen = []

    def run(command, **kwargs):
        seen.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout='{"results": [], "errors": []}')

    monkeypatch.setattr(evaluator.subprocess, "run", run)

    evaluator.run_batch_cli(dataset, args)

    assert seen[seen.index("--model") + 1] == "custom-model"
    if backend == "ollama":
        assert "--ollama=http://pop-os.local:11434/v1" in seen
        assert "--provider" not in seen
    else:
        assert seen[seen.index("--provider") + 1] == backend
        assert not any(arg.startswith("--ollama") for arg in seen)


def test_evaluator_ollama_runs_real_batch_path_and_reports_model(
    server, monkeypatch, tmp_path, capsys,
):
    requests, clients, _ = server
    monkeypatch.setattr(sys, "argv", [
        "eval_triage.py", "--ollama=pop-os.local", "--model", "qwen3.6:35b",
        "--request-interval", "0.001",
    ])
    monkeypatch.setattr(triage, "OUT_DIR", tmp_path / "out")

    def run(command, **kwargs):
        assert "--provider" not in command
        output = io.StringIO()
        with monkeypatch.context() as child:
            child.setattr(sys, "argv", command[1:])
            child.setattr(sys, "stdout", output)
            code = triage.main()
        return subprocess.CompletedProcess(command, code, stdout=output.getvalue())

    monkeypatch.setattr(evaluator.subprocess, "run", run)

    assert evaluator.main() == 0

    out, _ = capsys.readouterr()
    assert "Provider: Ollama" in out
    assert "qwen3.6:35b" in out
    assert "http://pop-os.local:11434/v1" in out
    assert len(requests) == 20
    assert all(json.loads(request.content)["model"] == "qwen3.6:35b" for request in requests)
    assert all(client.is_closed() for client in clients)


def test_ollama_calendar_proposals_use_existing_creation_workflow(server, monkeypatch, capsys):
    _, _, responses = server
    day = date(2030, 2, 1)
    result = triage.TriageAnalysis(
        category=triage.Category.other, priority=2, summary="Appointment.",
        extracted=triage.Extracted(dates=[day]),
        proposed_events=[triage.CalendarEvent(
            title="Appointment", start=day, end=date(2030, 2, 2), description="Discuss plans",
        )],
    )
    responses[:] = [(200, chat_response(content=result.model_dump_json()))]
    writes = []

    def create(calendar, title, start, end, description):
        writes.append((calendar, title, start, end, description))
        return {"id": "local-calendar-event"}

    monkeypatch.setattr(triage, "create_calendar_entry", create)
    monkeypatch.setattr(sys, "argv", [
        "triage.py", "--ollama=localhost", "--model", "qwen-test", "--json", "--create-events",
    ])
    monkeypatch.setattr(sys, "stdin", io.StringIO("An appointment"))

    assert triage.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["id"] == "***stdin***"
    assert output["calendar_events"][0]["id"] == "local-calendar-event"
    assert writes == [(triage.Calendar.google, "Appointment", day, date(2030, 2, 2), "Discuss plans")]
