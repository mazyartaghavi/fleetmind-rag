# Development Guide

This guide explains how to prepare a local development environment, run FleetMind-RAG, execute quality checks, and contribute changes through Git.

## Prerequisites

Install the following tools before working with the project:

* Git
* Python 3.12
* uv
* Visual Studio Code
* GitHub CLI

Verify the main tools:

```powershell
git --version
uv --version
gh --version
```

## Clone the Repository

Clone the public repository and enter its directory:

```powershell
git clone https://github.com/mazyartaghavi/fleetmind-rag.git
Set-Location fleetmind-rag
```

## Install Python and Project Dependencies

FleetMind-RAG uses Python 3.12 and `uv` for Python and dependency management.

Install Python 3.12 if necessary:

```powershell
uv python install 3.12
```

Create or synchronize the project environment:

```powershell
uv sync
```

This command:

* Reads `pyproject.toml`
* Uses the versions resolved in `uv.lock`
* Creates the `.venv` virtual environment when necessary
* Installs runtime and development dependencies
* Installs the local FleetMind-RAG package

## Open the Project in Visual Studio Code

From the repository root, run:

```powershell
code .
```

VS Code should detect and activate the project environment in its integrated terminal.

The terminal prompt should resemble:

```text
(fleetmind-rag) PS C:\path\to\fleetmind-rag>
```

## Run the Application

Run the current command-line application:

```powershell
uv run fleetmind-rag
```

During the initial foundation stage, the expected output is:

```text
Hello from fleetmind-rag!
```

## Local Configuration

The `.env.example` file documents the planned configuration variables.

Create a private local configuration file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` with local values when required.

Never commit:

* `.env`
* API keys
* Access tokens
* Passwords
* Private keys
* Database credentials

Git is configured to ignore `.env` and related local environment files while retaining `.env.example`.

## Project Structure

The current repository structure includes:

```text
fleetmind-rag/
├── docs/
│   └── development.md
├── src/
│   └── fleetmind_rag/
│       └── __init__.py
├── tests/
│   └── test_smoke.py
├── .editorconfig
├── .env.example
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

Application code belongs under `src/fleetmind_rag`, while automated tests belong under `tests`.

## Dependency Management

Add a runtime dependency with:

```powershell
uv add PACKAGE_NAME
```

Add a development-only dependency with:

```powershell
uv add --dev PACKAGE_NAME
```

After dependency changes, `uv` updates:

* `pyproject.toml`
* `uv.lock`
* The local virtual environment

Verify that the lockfile is current:

```powershell
uv lock --check
```

## Code Formatting

FleetMind-RAG uses Ruff as its Python formatter.

Check formatting without changing files:

```powershell
uv run ruff format --check .
```

Apply formatting:

```powershell
uv run ruff format .
```

Rerun the formatting check after automatic changes.

## Linting

Ruff also checks Python code for correctness, import ordering, common bugs, modernization opportunities, and simplification rules.

Run the linter:

```powershell
uv run ruff check .
```

Apply supported automatic fixes:

```powershell
uv run ruff check . --fix
```

Review all automatic changes before committing them.

## Static Type Checking

FleetMind-RAG uses mypy in strict mode.

Check application and test code:

```powershell
uv run mypy src tests
```

Type checking helps identify inconsistent function arguments, return values, attributes, and unreachable code without executing the application.

## Automated Tests

Run the test suite:

```powershell
uv run pytest
```

pytest discovers tests in the `tests` directory. Test files and test functions follow names such as:

```text
test_example.py
test_expected_behavior()
```

## Test Coverage

Measure which application statements and branches execute during testing:

```powershell
uv run pytest --cov=fleetmind_rag --cov-report=term-missing
```

The project currently requires at least 80 percent total coverage.

Coverage is useful for identifying untested code, but high coverage alone does not prove correctness. Tests should verify meaningful behavior and failure cases.

## Pre-commit Hooks

Install the repository’s Git pre-commit hook:

```powershell
uv run pre-commit install
```

Run every configured hook manually:

```powershell
uv run pre-commit run --all-files
```

The configured hooks check:

* Large added files
* Filename case conflicts
* Unresolved merge markers
* TOML syntax
* YAML syntax
* Final newlines
* Trailing whitespace
* Ruff linting
* Ruff formatting

A hook may modify a file and stop the first run. Review the modification and run the hooks again until all checks pass.

## Complete Local Quality Gate

Before creating a commit or pull request, run:

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest --cov=fleetmind_rag --cov-report=term-missing
uv run pre-commit run --all-files
```

All commands should pass.

## Git Branch Workflow

Start new work from an updated `main` branch:

```powershell
git switch main
git pull --ff-only origin main
```

Create a focused feature branch:

```powershell
git switch -c TYPE/SHORT-DESCRIPTION
```

Examples:

```text
feat/local-llama-client
feat/document-retrieval
fix/citation-validation
test/retrieval-evaluation
docs/architecture-guide
chore/python-quality-tooling
```

Inspect changes regularly:

```powershell
git status
git diff
```

Stage only intended files:

```powershell
git add FILE_1 FILE_2
```

Create a focused commit:

```powershell
git commit -m "type: concise description"
```

Push the branch and establish upstream tracking:

```powershell
git push -u origin BRANCH_NAME
```

Create a pull request with GitHub CLI or through the GitHub website.

## Commit Message Categories

Use clear commit prefixes:

```text
feat:     new application functionality
fix:      bug correction
docs:     documentation changes
test:     automated-test changes
refactor: internal restructuring without intended behavior changes
chore:    tooling, dependencies, or repository maintenance
perf:     performance improvement
```

## Generated Files

Do not commit generated local artifacts such as:

```text
.venv/
.coverage
htmlcov/
coverage.xml
.pytest_cache/
.mypy_cache/
.ruff_cache/
__pycache__/
```

These files can be recreated by the development tools and are excluded through `.gitignore`.

## Troubleshooting

### The project environment is missing

Run:

```powershell
uv sync
```

### The virtual environment is not active in VS Code

Open a new integrated terminal or select the interpreter located at:

```text
.venv\Scripts\python.exe
```

### Ruff reports formatting differences

Run:

```powershell
uv run ruff format .
```

Then check again:

```powershell
uv run ruff format --check .
```

### pytest reports that no tests were collected

Confirm that:

* The `tests` directory exists
* Test filenames begin with `test_`
* Test function names begin with `test_`

### A pre-commit hook modifies a file

Review the modification:

```powershell
git diff
```

Stage the corrected file and rerun:

```powershell
uv run pre-commit run --all-files
```

### Git reports that the current directory is not a repository

Return to the FleetMind-RAG root:

```powershell
Set-Location C:\path\to\fleetmind-rag
```

Then verify:

```powershell
git status
```
