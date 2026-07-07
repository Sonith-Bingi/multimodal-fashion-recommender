.PHONY: install test lint train evaluate docker-build docker-up

install:
	pip install -e ".[dev,ci]"

test:
	pytest

lint:
	ruff check .

train:
	reco train

evaluate:
	reco evaluate

docker-build:
	docker build -t fashion-recommender .

docker-up:
	docker compose up --build
