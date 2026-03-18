.PHONY: install lint test check summary

install:
	pip install -e .[dev]

lint:
	ruff check .

test:
	pytest

check:
	reco check

summary:
	reco summary
