# Contributing to Apsides Voice API

Thanks for your interest in improving Apsides Voice API. This guide covers local setup, project conventions, and how to submit changes.

## Ways to contribute

- **Bug reports** — open an issue with steps to reproduce, expected vs. actual behaviour, and your environment (OS, Python version, host).
- **Feature requests** — describe the use case and the outcome you want, not just the implementation.
- **Documentation** — fixes and clarifications to the README, `DEPLOYMENT.md`, or docstrings.
- **Code** — bug fixes, performance improvements, and new optional backends.

## Development setup

```bash
git clone https://github.com/apsideslabs/apsides-voice-api.git
cd apsides-voice-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
cp .env.example .env
uvicorn main:app --reload --port 8000
```

Run the end-to-end smoke test before opening a PR:

```bash
python scripts/verify_api.py            # against a running instance
API_KEY=xxxx python scripts/verify_api.py   # if auth is enabled
```

## Project conventions

- **Python 3.11–3.13.** Keep the dependency set minimal — this project deliberately targets constrained hosts.
- **No secrets in code or commits.** Every credential (e.g. `API_KEY`) is read from the environment only. Never commit a `.env`.
- **Resource discipline.** New work must respect the CPU-only, low-memory budget: keep inference bounded, avoid unbounded concurrency, and prefer lazy loading.
- **Config over constants.** New tunables should be environment variables wired through `app/config.py` and documented in `.env.example`.
- **Failure is graceful.** A missing or broken model must not stop the service from booting; surface the reason via `/health`.
- **Type hints and docstrings** on public functions and modules.

## Pull request checklist

- [ ] The change is focused and described clearly in the PR body.
- [ ] `python scripts/verify_api.py` passes (`RESULT: PASS`).
- [ ] New configuration is added to `.env.example` and the README.
- [ ] No secrets, large model files, or generated artifacts are committed (see `.gitignore`).
- [ ] Documentation is updated where behaviour changed.

## Commit messages

Use short, imperative subjects (e.g. `Add aarch64 fallback for transcribe bundle`). Reference the relevant issue where applicable.

## Code of conduct

Be respectful and constructive. Assume good faith, keep discussion technical, and help newcomers.

## Questions

Open a discussion or an issue on GitHub, or reach the maintainers at **apsideslabs@gmail.com**.
