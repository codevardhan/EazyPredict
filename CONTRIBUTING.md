# Contributing

## Development environment

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv sync --all-extras --group dev
```

Python 3.12 is pinned in `.python-version`. Do not build the dev environment against a
very new interpreter (e.g. 3.14): it may ship without `pip`, and `xgboost`/`lightgbm`
wheels typically lag several months behind a new CPython release.

## Checks

```bash
uv run pytest                  # fast suite
uv run pytest --runslow        # includes booster and conformance tests
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Install the hooks so these run automatically:

```bash
uv run pre-commit install
```

## Testing conventions

- Keep fixtures tiny (`n_samples <= 80`, `n_features <= 6`). The default suite should
  stay under about 20 seconds.
- Tests that fit the full default estimator list, run hyperparameter search, or invoke
  scikit-learn's `check_estimator` must be marked `@pytest.mark.slow`.
- Every bug fix gets a named regression test in `tests/test_regressions.py`.
