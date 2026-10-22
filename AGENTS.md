# AGENTS.md

## Project

Python 3.14 application.

## Development

Run the full test suite from VS Code with **Tasks: Run Test Task** or the
`Railbound: Run all tests` task.

Agents must not run `pytest` or invoke the VS Code test task. Ask the user to
run it and report its result instead.

Run linting with:

    ruff check .

Run formatting with:

    ruff format .

Run type checking with:

    mypy src

## Code style

- Use Python 3.14 syntax.
- Add type annotations to public functions.
- Prefer pathlib over os.path.
- Keep functions small.
- Use Python tooling to apply changes from inside the devcontainer.
- Do not create temporary files or backup copies, including files with `.bak` or `.orig` suffixes.
- Do not modify tests merely to make failing code pass.
- Add tests for new behavior.

## Before completing a task

Run:

    ruff check .

Do not run tests. Remind the user to use `Railbound: Run all tests` if test
verification is needed.
