.PHONY: test lint coverage build-binaries install clean

test:
	pytest tests/ -v --tb=short

lint:
	ruff check catboost_mlx/ tests/

coverage:
	pytest tests/ -v --tb=short --cov=catboost_mlx --cov-report=term-missing

build-binaries:
	python build_binaries.py

install:
	pip install -e ".[dev]"

clean:
	rm -rf build/ dist/ *.egg-info catboost_mlx/__pycache__ tests/__pycache__ .pytest_cache
