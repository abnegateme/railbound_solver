# AGENTS.md

## Project

Python 3.14 application.

## Development

The user runs project verification from VS Code:

- **Tasks: Run Test Task** or `Railbound: Run all tests` runs the full test suite.
- `Railbound: Verify changes` runs linting, type checking, and diff validation.

Agents must not run either VS Code task or their underlying commands: `pytest`,
`ruff check .`, `mypy src`, or `git diff --check`. Ask the user to run the
relevant task and report its result when verification is needed.

Run formatting with:

    ruff format .

Run type checking with:

    mypy src

## Code style

- Use Python 3.14 syntax.
- Add type annotations to public functions.
- Prefer pathlib over os.path.
- Keep functions small.
- Keep documentation concise and concrete. Do not overcomplicate wording,
  repeat information, or add unnecessary detail and filler.
- Use Python tooling to apply changes from inside the devcontainer.
- Do not create temporary files or backup copies, including files with `.bak` or `.orig` suffixes.
- Do not modify tests merely to make failing code pass.
- Add tests for new behavior.

## Before completing a task

Do not run verification commands. Remind the user to use the relevant VS Code
task when test or change verification is needed.
