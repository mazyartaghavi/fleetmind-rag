import pytest

from fleetmind_rag import main


def test_main_prints_expected_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main()

    captured = capsys.readouterr()

    assert captured.out == "Hello from fleetmind-rag!\n"
    assert captured.err == ""
