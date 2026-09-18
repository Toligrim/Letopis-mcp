# Repository Guidelines for Coding Agents

This document provides guidelines for coding agents (e.g., Jules, Codex, Claude Code, and other AI tools) working in this repository.

## 1. Sensitive and Data Files Protection

Never commit, expose, or accidentally modify sensitive configuration or data files:
- `config.toml` (contains API keys and application secrets; use `config.example.toml` for schema changes)
- `.env`
- `data/` (derived SQLite indices and internal database files)
- `archive/` (raw chat message logs and media files)
- `*.session` (Telegram session files)

Ensure git status and untracked files are inspected before creating commits.

## 2. Read-Only MCP Architecture

The Model Context Protocol (MCP) layer in this repository is **read-only by design**.
- Do not introduce archive mutations, Telegram sync/download operations, destructive database modifications, or write APIs into the MCP layer unless explicitly requested in a future task.
- Treat `data/index.db` as a read-only database in MCP operations.

## 3. Code Architecture & Public Contracts

- MCP server implementation code resides under `tgarchive/mcp/`.
- Maintain clean separation of concerns and avoid unnecessary coupling between `tgarchive/mcp/` and unrelated CLI/engine internals.
- Do not casually modify public tool contracts, tool output shapes, JSON schemas, or runtime contracts without explicit direction.

## 4. Development Environment & Dependencies

- Use `uv` and `pyproject.toml` for environment setup and dependency management.
- Development dependencies should be installed with `uv` (e.g., `uv run --extra mcp --extra test pytest tests/mcp/`).
- Do not add new external dependencies unless required and explicitly justified.

## 5. Testing Expectations

- Always run relevant tests when modifying code.
- Run the MCP test suite using `uv run --extra mcp --extra test pytest tests/mcp/`.
- Ensure changes pass existing test suites without causing regressions.

## 6. Git Workflow Rules

- Always work on a separate, dedicated topic branch.
- Do not commit directly to `master`.
- Do not merge pull requests yourself.
- Keep changes focused, concise, and aligned strictly with the task instructions.
