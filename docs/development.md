# Development Guide

This guide covers local setup, quality checks, testing, and the Git workflow
for FleetMind-RAG.

## Prerequisites

Install Git, Python 3.12, uv, Visual Studio Code, GitHub CLI, and Ollama.

Verify them in the VS Code integrated PowerShell terminal:

```powershell
git --version
uv --version
gh --version
ollama --version
```

## Environment Setup

Clone the project and enter its root:

```powershell
git clone https://github.com/mazyartaghavi/fleetmind-rag.git
Set-Location fleetmind-rag
```

Install Python and synchronize exactly from the lockfile:

```powershell
uv python install 3.12
uv sync --locked
```

Create private local configuration:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`, credentials, private feedback, Qdrant local data, model
files, or generated caches.

## Local Models

The default configuration expects:

```powershell
ollama pull llama3.2:3b
ollama pull embeddinggemma
ollama list
```

Check the configured runtime:

```powershell
uv run fleetmind-rag
```

## Source Layout

Application modules live in `src/fleetmind_rag`; tests mirror them under
`tests`.

Key module groups:

- document ingestion and chunking: `documents.py`
- Ollama clients: `ollama.py`
- vector persistence: `vector_store.py`
- dense, sparse, hybrid, and reranked search: `retrieval.py`
- query routing and execution: `routing.py`, `routed_retrieval.py`
- adaptive state and retries: `agent_state.py`, `adaptive_retrieval.py`
- quality checks: `retrieval_quality.py`
- LangGraph workflow: `langgraph_workflow.py`
- grounded generation: `grounded_rag.py`, `adaptive_grounded_rag.py`
- feedback control: `feedback_routing.py`, `feedback_store.py`,
  `feedback_analytics.py`, `feedback_trends.py`, `feedback_gates.py`
- runtime and CLI composition: `runtime.py`, `app.py`

## Dependency Management

Add a runtime dependency:

```powershell
uv add PACKAGE_NAME
```

Add a development-only dependency:

```powershell
uv add --dev PACKAGE_NAME
```

Validate dependency resolution:

```powershell
uv lock --check
```

Commit both `pyproject.toml` and `uv.lock` when a dependency intentionally
changes.

## Formatting and Linting

Check formatting:

```powershell
uv run ruff format --check .
```

Apply formatting:

```powershell
uv run ruff format .
```

Run linting:

```powershell
uv run ruff check .
```

Apply supported lint fixes:

```powershell
uv run ruff check . --fix
```

Review automatic changes before staging them.

## Strict Type Checking

```powershell
uv run mypy src tests
```

The project uses strict mypy settings and checks application code and tests.

## Tests and Coverage

Run the complete suite:

```powershell
uv run python -m pytest -q
```

Run coverage:

```powershell
uv run pytest `
    --cov=fleetmind_rag `
    --cov-report=term-missing
```

The configured project-wide minimum is 80 percent. Coverage complements, but
does not replace, behavioral assertions and failure-case tests.

## Compilation and Whitespace

```powershell
uv run python -m compileall `
    src tests `
    -q

git diff --check
```

Both commands should be quiet when successful.

## Pre-commit Hooks

Install the hook once:

```powershell
uv run pre-commit install
```

Run every hook:

```powershell
uv run pre-commit run --all-files
```

Hooks check large files, filename conflicts, merge markers, TOML and YAML,
final newlines, trailing whitespace, Ruff linting, and Ruff formatting.

## Complete Local Quality Gate

Before committing:

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest --cov=fleetmind_rag --cov-report=term-missing
uv run python -m compileall src tests -q
git diff --check
uv run pre-commit run --all-files
```

## Git Workflow

Start from synchronized `main`:

```powershell
git switch main
git pull --ff-only
git status
```

Create a focused branch:

```powershell
git switch -c TYPE/SHORT-DESCRIPTION
```

Common prefixes:

| Prefix | Purpose |
| --- | --- |
| `feat` | New application behavior |
| `fix` | Defect correction |
| `test` | Test or evaluation work |
| `docs` | Documentation |
| `ci` | Continuous-integration changes |
| `refactor` | Internal change without intended behavior change |
| `chore` | Maintenance or tooling |

Inspect and stage only intended paths:

```powershell
git status --short
git diff
git add FILE_1 FILE_2
git diff --cached --check
```

Commit, push, and create a pull request:

```powershell
git commit -m "type: concise description"
git push -u origin BRANCH_NAME
gh pr create --base main --head BRANCH_NAME
```

Wait for checks before merging:

```powershell
gh pr checks --watch
gh pr view
```

After review:

```powershell
gh pr merge --squash --delete-branch
git switch main
git pull --ff-only
git status
```

## GitHub Actions

`.github/workflows/ci.yml` runs on pull requests to `main` and pushes to
`main`. It performs:

1. Locked dependency installation.
2. Formatting.
3. Linting.
4. Strict type checking.
5. Tests with coverage.
6. Strict routing-feedback regression gating.

The gate reads `evaluation/data/routing_feedback_ci.json`, not local runtime
feedback.

## Generated and Private Files

Do not commit:

```text
.env
.venv/
.coverage
htmlcov/
coverage.xml
.pytest_cache/
.mypy_cache/
.ruff_cache/
__pycache__/
data/qdrant_local/
```

## Troubleshooting

### Environment is missing or inconsistent

```powershell
uv sync --locked
```

### VS Code does not activate the environment

Select:

```text
.venv\Scripts\python.exe
```

Then open a new integrated terminal.

### Formatting check fails

```powershell
uv run ruff format .
uv run ruff format --check .
```

### A hook changes files

```powershell
git diff
uv run pre-commit run --all-files
```

Restage the corrected files before committing.

For runtime and feedback problems, see
[`operations.md`](operations.md).
