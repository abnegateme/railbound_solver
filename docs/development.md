# Development and runtime

## Devcontainer

The supported development environment is the Python 3.14 devcontainer. The
checkout is bind-mounted and exposed through `PYTHONPATH`; it is not installed
into that environment. Rebuild the devcontainer after changing its definition.

List the available stage commands with:

```bash
python3 -m railbound --help
```

## Checks

Use the VS Code tasks:

- `Railbound: Verify changes` for linting, type checking, and diff validation.
- `Railbound: Run all tests` for the full test suite.

## Headless release image

The Dockerfile's `release` target installs the package non-editably with core
dependencies into `/opt/railbound-venv`, without development tools or the
source checkout:

```bash
docker build -f .devcontainer/Dockerfile --target release -t railbound-solver .
docker run --rm railbound-solver --help
```
