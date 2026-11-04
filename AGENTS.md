# AGENTS.md

## Project

Python 3.14 application.

## File access

- Read only files located directly in the repository root and files under
  `src/` and `docs/`.
- Read files in any other directory, or files outside those locations, only
  when the user explicitly identifies the file or directory to inspect.

## Token efficiency

- Use only the context and tool output needed to solve the current task.
- Prefer targeted searches and partial file reads over broad scans, avoid
  rereading unchanged content, and keep explanations concise.
- Do not trade away correctness, required verification, or essential context
  merely to reduce token usage.

## Development

The user runs project verification from VS Code:

- **Tasks: Run Test Task** or `Railbound: Run all tests` runs the full test suite.
- `Railbound: Verify changes` runs linting, type checking, and diff validation.

Agents must not run either VS Code task. They also must not run `ruff check .`,
`mypy src`, or `git diff --check`. Do not create or run tests, including through
`pytest`, unless the user explicitly asks for that action. Otherwise, ask the
user to run the relevant task and report its result when verification is needed.

Run formatting with:

    ruff format .

Run type checking with:

    mypy src

## Code style

- Use Python 3.14 syntax.
- Add type annotations to public functions.
- Prefer pathlib over os.path.
- Keep functions small.
- Keep documentation compact without losing essential detail; prefer short
  files and split documents when that improves clarity.
- Use Python tooling to apply changes from inside the devcontainer.
- Do not create temporary files or backup copies, including files with `.bak` or `.orig` suffixes.
- Do not modify tests merely to make failing code pass.

## Before completing a task

Do not run verification commands. Remind the user to use the relevant VS Code
task when test or change verification is needed.
