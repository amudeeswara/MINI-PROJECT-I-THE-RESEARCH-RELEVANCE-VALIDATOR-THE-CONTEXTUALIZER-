# Contributing

1. Fork and create a branch.
2. `pip install -e ".[dev]"` then `pytest -q` — all tests must pass.
3. Add a test for every behaviour change. If you change scoring, re-check `contextualizer demo` still ranks ref-01 > ref-02 > the rest.
4. Keep the runtime dependency-free; heavier features belong in optional extras.
5. Open a pull request describing the motivation.
