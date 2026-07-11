# FleetMind-RAG

An agentic retrieval-augmented generation system for intelligent fleet operations, technical-document assistance, and feedback-driven retrieval-policy optimization.

> **Project status:** Foundation stage. The repository is being developed incrementally from Python engineering fundamentals to production-oriented RAG, agentic workflows, evaluation, and reinforcement-learning-based retrieval routing.

## Overview

FleetMind-RAG is a project-based learning and portfolio system that combines generative AI with reinforcement learning.

The completed system will assist fleet operators with tasks such as:

* Retrieving evidence from maintenance manuals and operational procedures
* Answering technical questions with document citations
* Inspecting simulated vehicle telemetry and maintenance history
* Recommending operational or maintenance actions
* Preparing maintenance tickets through controlled tool calls
* Escalating uncertain or safety-sensitive decisions for human review
* Learning which retrieval strategy performs best for different queries

The project is designed as a production-oriented AI application rather than a standalone chatbot.

## Project Objectives

The project has four main objectives:

1. Build a reliable retrieval-augmented generation pipeline.
2. Develop a stateful, tool-using agentic workflow.
3. Evaluate retrieval, generation, safety, latency, and cost quantitatively.
4. Apply contextual-bandit reinforcement learning to retrieval-strategy selection.

## Planned System Flow

```text
User
  |
  v
API or user interface
  |
  v
Agentic workflow
  |
  +----> Document retrieval and reranking
  |
  +----> Fleet telemetry and maintenance tools
  |
  +----> Local or hosted LLM
  |
  v
Grounded answer, citations, or human escalation
  |
  v
Evaluation and user feedback
  |
  v
Contextual-bandit retrieval router
```

## Development Roadmap

* [x] Initialize the Python package and Git repository
* [x] Configure Python 3.12 and dependency management with `uv`
* [x] Publish the initial public GitHub repository
* [ ] Add code-quality, type-checking, testing, and CI tools
* [ ] Run a local Llama model through Ollama
* [ ] Implement structured prompting and validated model output
* [ ] Build document ingestion, chunking, and embeddings
* [ ] Add Qdrant semantic search
* [ ] Implement basic and advanced RAG
* [ ] Refactor the pipeline with LangChain
* [ ] Build a stateful LangGraph workflow
* [ ] Add fleet tools and human approval
* [ ] Create retrieval and generation evaluation suites
* [ ] Implement a contextual-bandit retrieval router
* [ ] Expose the application through FastAPI
* [ ] Add Docker, observability, security checks, and CI/CD

## Planned Technology Stack

| Area                         | Technology                            |
| ---------------------------- | ------------------------------------- |
| Programming language         | Python 3.12                           |
| Project management           | uv                                    |
| Local language model         | Llama through Ollama                  |
| LLM application framework    | LangChain                             |
| Agent orchestration          | LangGraph                             |
| Vector database              | Qdrant                                |
| API                          | FastAPI                               |
| Validation and configuration | Pydantic                              |
| Testing                      | pytest                                |
| Linting and formatting       | Ruff                                  |
| Static type checking         | mypy                                  |
| Containers                   | Docker and Docker Compose             |
| Continuous integration       | GitHub Actions                        |
| Reinforcement learning       | Contextual bandits                    |
| Evaluation and observability | Custom metrics and optional LangSmith |

Technology entries marked as planned will be introduced progressively as the corresponding project stages are implemented.

## Quick Start

### Prerequisites

Install:

* Git
* Python 3.12
* uv

### Clone the Repository

```powershell
git clone https://github.com/mazyartaghavi/fleetmind-rag.git
Set-Location fleetmind-rag
```

### Install the Environment

```powershell
uv python install 3.12
uv sync
```

### Run the Current Application

```powershell
uv run fleetmind-rag
```

During the foundation stage, the expected output is:

```text
Hello from fleetmind-rag!
```

## Configuration

The repository contains `.env.example`, which documents the planned environment variables.

Create a local configuration file with:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`, access tokens, API keys, passwords, or other credentials.

## Current Repository Structure

```text
fleetmind-rag/
├── docs/
│   └── development.md
├── src/
│   └── fleetmind_rag/
│       └── __init__.py
├── .editorconfig
├── .env.example
├── .gitattributes
├── .gitignore
├── .python-version
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

The repository structure will expand as retrieval, agents, evaluation, APIs, and reinforcement learning are implemented.

## Evaluation Plan

The completed project will evaluate:

* Retrieval recall and ranking quality
* Answer relevance and faithfulness
* Citation correctness and completeness
* Abstention accuracy
* Tool-call success
* Human-escalation behavior
* End-to-end latency
* Computational cost
* Retrieval-policy reward

Experimental results will be generated from reproducible evaluation scripts rather than manually entered values.

## Responsible AI Principles

FleetMind-RAG will be designed to:

* Ground operational recommendations in retrieved evidence
* Display document sources
* Abstain when evidence is insufficient
* Separate read-only tools from state-changing actions
* Require human approval for sensitive actions
* Record evaluation and failure cases
* Avoid presenting the system as a substitute for qualified technical or safety personnel

## Documentation

Development setup instructions are available in [`docs/development.md`](docs/development.md).

Additional architecture, RAG, agent, evaluation, and reinforcement-learning documentation will be added as the project develops.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Mazyar Taghavi**

AI engineer and reinforcement-learning researcher developing practical systems at the intersection of reinforcement learning, generative AI, intelligent agents, and mathematical optimization.
