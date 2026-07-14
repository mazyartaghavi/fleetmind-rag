from pathlib import Path

import pytest

from fleetmind_rag.app import main
from fleetmind_rag.ollama import OllamaChatResult


def test_chat_command_prints_model_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeOllamaChatClient:
        def __init__(
            self,
            base_url: str,
            model: str,
        ) -> None:
            assert base_url == "http://localhost:11434/"
            assert model == "llama3.2:3b"

        def chat(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> OllamaChatResult:
            assert prompt == "Summarize fleet status."
            assert system_prompt == "Answer briefly."

            return OllamaChatResult(
                succeeded=True,
                content="All monitored vehicles are operational.",
                model="llama3.2:3b",
                message="The Ollama chat request succeeded.",
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fleetmind_rag.app.OllamaChatClient",
        _FakeOllamaChatClient,
    )

    exit_code = main(
        [
            "chat",
            "Summarize fleet status.",
            "--system",
            "Answer briefly.",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "All monitored vehicles are operational.\n"
    assert captured.err == ""


def test_chat_command_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeFailedOllamaChatClient:
        def __init__(
            self,
            base_url: str,
            model: str,
        ) -> None:
            assert base_url == "http://localhost:11434/"
            assert model == "llama3.2:3b"

        def chat(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> OllamaChatResult:
            assert prompt == "Hello"
            assert system_prompt is None

            return OllamaChatResult(
                succeeded=False,
                content=None,
                model=None,
                message="The Ollama API is unreachable.",
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fleetmind_rag.app.OllamaChatClient",
        _FakeFailedOllamaChatClient,
    )

    exit_code = main(["chat", "Hello"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == ("FleetMind chat failed: The Ollama API is unreachable.\n")
