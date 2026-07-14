from pathlib import Path

import pytest

from fleetmind_rag.app import (
    DEFAULT_SYSTEM_PROMPT,
    build_system_prompt,
    main,
)
from fleetmind_rag.ollama import OllamaChatResult


@pytest.mark.parametrize(
    "user_system_prompt",
    [
        None,
        "",
        "   ",
    ],
)
def test_build_system_prompt_uses_default_for_missing_instruction(
    user_system_prompt: str | None,
) -> None:
    result = build_system_prompt(user_system_prompt)

    assert result == DEFAULT_SYSTEM_PROMPT


def test_build_system_prompt_appends_user_instruction() -> None:
    result = build_system_prompt("  Answer briefly.  ")

    assert result == (
        f"{DEFAULT_SYSTEM_PROMPT}\n\nAdditional user instruction: Answer briefly."
    )


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
            assert system_prompt == (
                f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                "Additional user instruction: Answer briefly."
            )

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
            assert system_prompt == DEFAULT_SYSTEM_PROMPT

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
