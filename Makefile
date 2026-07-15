.PHONY: help install fixtures fixtures-domains fixtures-minds14 clean-fixtures showcase smoke clean

VENV_PY := .venv/bin/python
VENV_BIN := .venv/bin

help:
	@echo "gemma-stt Makefile targets:"
	@echo "  install           uv venv + uv pip install -e ."
	@echo "  fixtures          Download all test fixtures (domains + minds14)"
	@echo "  fixtures-domains  Download legal/medical/financial clips (curl+ffmpeg, no extra deps)"
	@echo "  fixtures-minds14  Download E2B/E4B comparison clips (installs the 'fixtures' extra: HF datasets)"
	@echo "  showcase          Run the domain-prompting comparison (needs fixtures-domains + --model e4b by default)"
	@echo "  smoke             Quick end-to-end sanity check (needs fixtures-minds14)"
	@echo "  clean-fixtures    Delete all downloaded fixture .wav files"
	@echo "  clean             Remove .venv, caches, and fixture audio"

install:
	uv venv
	uv pip install -e .

fixtures: fixtures-domains fixtures-minds14

fixtures-domains:
	$(VENV_PY) scripts/fetch_fixtures.py domains

fixtures-minds14:
	uv pip install -e ".[fixtures]"
	$(VENV_PY) scripts/fetch_fixtures.py minds14

showcase: fixtures-domains
	$(VENV_PY) tests/run_domain_showcase.py --model e4b

smoke: fixtures-minds14
	$(VENV_BIN)/gemma-stt transcribe tests/fixtures/sample_5.wav --model e2b

clean-fixtures:
	find tests/fixtures -name '*.wav' -delete

clean: clean-fixtures
	rm -rf .venv
	find . -name '__pycache__' -type d -exec rm -rf {} +
