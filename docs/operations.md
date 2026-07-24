# Operations Runbook

This runbook covers local startup, indexing, adaptive answering, feedback
inspection, regression gates, recovery, and common failures.

## Runtime Requirements

- Python 3.12 environment synchronized with `uv sync --locked`
- Ollama reachable at the configured base URL
- configured chat and embedding models installed
- writable Qdrant local directory

Default local paths and model names are documented in `.env.example`.

## Startup Check

```powershell
ollama list
uv run fleetmind-rag
```

The FleetMind status command reports active configuration, Ollama health, and
available models. A missing model should be pulled before indexing or asking
questions.

## Indexing

Create or replace the local evaluation collection:

```powershell
uv run fleetmind-rag index `
    evaluation/data/fleet_manual.md `
    --recreate
```

Use `--recreate` only when replacing the current collection is intended.

After indexing, the command reports document, section, chunk, stored-vector,
embedding-model, and vector-size information.

## Grounded Answer Smoke Test

```powershell
uv run fleetmind-rag ask `
    "What should the driver do if a battery warning appears with smoke?" `
    --adaptive `
    --limit 5 `
    --max-attempts 3 `
    --candidate-limit 20
```

Confirm:

- the answer contains source labels;
- safety conditions and prohibited actions are explicit;
- adaptive status is completed;
- attempt count does not exceed the configured limit;
- a feedback revision is reported.

## Persistent Feedback

Runtime feedback is stored at:

```text
data/qdrant_local/routing_feedback.json
```

This file is private runtime state and is ignored by Git.

Read the current snapshot:

```powershell
uv run fleetmind-rag feedback-report
```

The command is read-only and does not start Ollama or query Qdrant.

## Trend Analysis

```powershell
uv run fleetmind-rag feedback-trend `
    --window-size 10 `
    --minimum-change 0.05 `
    --minimum-strategy-observations 2
```

Interpretation:

| Direction | Meaning |
| --- | --- |
| `improving` | Recent blended utility increased beyond the threshold |
| `stable` | Change remains inside the configured tolerance |
| `regressing` | Recent blended utility decreased beyond the threshold |
| `insufficient_data` | One or both windows lack required evidence |

## Regression Gate

Human-readable local check:

```powershell
uv run fleetmind-rag feedback-gate
```

Automation output:

```powershell
uv run fleetmind-rag feedback-gate `
    --format json `
    --fail-on fail
```

Enforcement modes:

| Mode | Nonzero statuses |
| --- | --- |
| `never` | None |
| `fail` | Fail only |
| `warn` | Warn and fail |

Process codes:

| Code | Operational meaning |
| ---: | --- |
| 0 | Allowed by enforcement |
| 1 | Command, configuration, or storage error |
| 2 | Enforced warning |
| 3 | Enforced regression |

In PowerShell, capture the gate code immediately:

```powershell
uv run fleetmind-rag feedback-gate `
    --format json `
    --fail-on warn

$gateExitCode = $LASTEXITCODE
$gateExitCode
```

Code `2` or `3` is an intentional gate signal, not necessarily a program
defect.

## CI Gate

GitHub Actions evaluates:

```text
evaluation/data/routing_feedback_ci.json
```

Reproduce it locally:

```powershell
uv run fleetmind-rag feedback-gate `
    --feedback-path evaluation/data/routing_feedback_ci.json `
    --window-size 10 `
    --minimum-change 0.05 `
    --minimum-strategy-observations 2 `
    --format json `
    --fail-on warn
```

Expected status is `pass`, direction is `stable`, and exit code is `0`.

Do not replace the fixture with private runtime data. Intentional fixture
changes require review because they change the CI policy input.

## Backup

Before maintenance, copy the local data directory while FleetMind operations
are stopped:

```powershell
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

Copy-Item `
    .\data\qdrant_local `
    ".\data\qdrant_local-backup-$timestamp" `
    -Recurse
```

Keep backups outside the Git staging area. The local data directory and its
backups should remain ignored.

## Recovery

If the feedback JSON is damaged:

1. Stop commands that may write feedback.
2. Preserve a copy of the damaged file for investigation.
3. Restore a known-good backup.
4. Run `feedback-report`.
5. Run the adaptive smoke test.

Do not manually change `revision` without understanding optimistic concurrency
checks.

If the Qdrant collection can be rebuilt from source documents, reindexing with
`--recreate` is preferable to editing database files manually.

## Failure Guide

### Ollama is unreachable

Check the process and configured URL:

```powershell
ollama list
uv run fleetmind-rag
```

Start Ollama and retry.

### Required model is missing

```powershell
ollama pull llama3.2:3b
ollama pull embeddinggemma
```

Use the names configured in `.env` if they differ.

### No retrieval matches

Confirm that:

- the document was indexed;
- the active collection name is correct;
- the Qdrant path matches the indexed environment;
- restrictive metadata filters are not excluding evidence.

### Feedback gate warns

Inspect:

```powershell
uv run fleetmind-rag feedback-report
uv run fleetmind-rag feedback-trend
```

A warning commonly means insufficient observations rather than degradation.

### Feedback gate fails

Review the overall direction, regressing strategies, utility delta, window
positions, and gate reasons. Do not weaken thresholds merely to make CI green;
determine whether the fixture, policy, analyzer, or retrieval behavior changed
intentionally.

### Feedback store conflict

Another process updated the snapshot after it was loaded. Reload the current
snapshot and retry the operation rather than overwriting the newer revision.

### Lock timeout

Confirm that no FleetMind process is actively writing. A stale lock may be
recovered by the store’s stale-lock handling; do not delete locks while a
writer is active.

## Release Checklist

Before declaring a revision ready:

```powershell
git status
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest --cov=fleetmind_rag --cov-report=term-missing
uv run fleetmind-rag feedback-gate `
    --feedback-path evaluation/data/routing_feedback_ci.json `
    --format json `
    --fail-on warn
```

Then verify that pull-request checks pass before squash-merging.
