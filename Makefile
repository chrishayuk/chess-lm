.PHONY: help install dev test test-verbose coverage coverage-html lint format type-check clean build all check

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := uv run python
PIP := uv pip
PYTEST := uv run pytest
BLACK := uv run black
RUFF := uv run ruff
MYPY := uv run mypy
ISORT := uv run isort

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Chess-LM Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make install         Install the package"
	@echo "  make test           Run tests"
	@echo "  make check          Run all checks (lint, type, test)"

install: ## Install the package in editable mode
	@echo "$(GREEN)Installing chess-lm...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"

dev: ## Install package with dev dependencies
	@echo "$(GREEN)Installing chess-lm with dev dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Dev installation complete$(NC)"

test: ## Run tests
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTEST) tests/ -q
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-verbose: ## Run tests with verbose output
	@echo "$(GREEN)Running tests (verbose)...$(NC)"
	$(PYTEST) tests/ -v

test-fast: ## Run tests in parallel (requires pytest-xdist)
	@echo "$(GREEN)Running tests in parallel...$(NC)"
	$(PYTEST) tests/ -n auto

coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ --cov=chess_lm --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage complete$(NC)"

coverage-html: ## Generate HTML coverage report
	@echo "$(GREEN)Generating HTML coverage report...$(NC)"
	$(PYTEST) tests/ --cov=chess_lm --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ HTML report generated in htmlcov/$(NC)"
	@echo "$(YELLOW)  Open htmlcov/index.html in your browser$(NC)"

lint: ## Run linting with ruff
	@echo "$(GREEN)Running ruff linter...$(NC)"
	$(RUFF) check src/ tests/ || (echo "$(RED)✗ Linting failed$(NC)" && exit 1)
	@echo "$(GREEN)✓ Linting passed$(NC)"

lint-fix: ## Auto-fix linting issues
	@echo "$(GREEN)Auto-fixing linting issues...$(NC)"
	$(RUFF) check --fix src/ tests/
	@echo "$(GREEN)✓ Auto-fix complete$(NC)"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/
	@echo "$(GREEN)✓ Formatting complete$(NC)"

format-check: ## Check code formatting without changes
	@echo "$(GREEN)Checking code format...$(NC)"
	$(BLACK) --check src/ tests/ || (echo "$(RED)✗ Formatting check failed$(NC)" && exit 1)
	$(ISORT) --check-only src/ tests/ || (echo "$(RED)✗ Import sorting check failed$(NC)" && exit 1)
	@echo "$(GREEN)✓ Format check passed$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checker...$(NC)"
	$(MYPY) src/ || true
	@echo "$(YELLOW)Note: Type checking is informational$(NC)"

clean: ## Clean build artifacts and cache
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✓ Clean complete$(NC)"

build: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Build complete$(NC)"
	@ls -lh dist/

# Composite targets

check: lint format-check test ## Run all checks (lint, format, test)
	@echo "$(GREEN)✅ All checks passed!$(NC)"

all: clean install lint format test coverage ## Run everything (clean, install, lint, format, test, coverage)
	@echo "$(GREEN)✅ All tasks completed!$(NC)"

# Development workflow commands

watch-tests: ## Watch files and re-run tests on changes (requires pytest-watch)
	@echo "$(GREEN)Watching for changes...$(NC)"
	$(PYTHON) -m pytest_watch tests/ --clear

# CLI testing commands

cli-test: ## Test the chess-tokenizer CLI
	@echo "$(GREEN)Testing chess-tokenizer CLI...$(NC)"
	chess-tokenizer --help
	chess-tokenizer info
	chess-tokenizer encode-seq --seq "e2e4 e7e5"
	@echo "$(GREEN)✓ CLI test complete$(NC)"

# Script testing commands

test-scripts: ## Test all scripts in scripts/
	@echo "$(GREEN)Testing scripts...$(NC)"
	$(PYTHON) scripts/encode_smoke.py
	$(PYTHON) scripts/generate_synthetic_games.py --out data/test_games.jsonl --games 2 --min-plies 10 --max-plies 20
	@echo "$(GREEN)✓ Scripts test complete$(NC)"

# Data generation commands

generate-data: ## Generate synthetic test data
	@echo "$(GREEN)Generating synthetic data...$(NC)"
	@mkdir -p data
	$(PYTHON) scripts/generate_synthetic_games.py \
		--out data/synthetic_games.jsonl \
		--games 100 \
		--min-plies 20 \
		--max-plies 50 \
		--seed 42
	@echo "$(GREEN)✓ Generated 100 games in data/synthetic_games.jsonl$(NC)"

# Documentation commands

docs: ## Generate documentation (if using sphinx/mkdocs)
	@echo "$(YELLOW)Documentation generation not configured yet$(NC)"

# Dependency management

deps-update: ## Update dependencies to latest versions
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) list --outdated
	@echo "$(YELLOW)Run 'uv pip install --upgrade <package>' to update specific packages$(NC)"

deps-check: ## Check for dependency security issues
	@echo "$(GREEN)Checking dependencies...$(NC)"
	$(PIP) list
	@echo "$(YELLOW)Consider using 'pip-audit' for security scanning$(NC)"

# Git hooks

install-hooks: ## Install pre-commit hooks
	@echo "$(GREEN)Installing git hooks...$(NC)"
	@echo "#!/bin/sh" > .git/hooks/pre-commit
	@echo "make lint" >> .git/hooks/pre-commit
	@echo "make test" >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "$(GREEN)✓ Pre-commit hook installed$(NC)"

# Performance profiling

profile: ## Run tests with profiling
	@echo "$(GREEN)Running tests with profiling...$(NC)"
	$(PYTEST) tests/ --profile --profile-svg
	@echo "$(GREEN)✓ Profile generated as prof/combined.svg$(NC)"

# Release commands

version: ## Show current version
	@$(PYTHON) -c "import chess_lm; print(f'chess-lm version: {chess_lm.__version__ if hasattr(chess_lm, \"__version__\") else \"0.1.0\"}')"

release-check: clean lint format-check test coverage build ## Check if ready for release
	@echo "$(GREEN)✅ Ready for release!$(NC)"
	@echo "$(YELLOW)Don't forget to:$(NC)"
	@echo "  1. Update version in pyproject.toml"
	@echo "  2. Update CHANGELOG.md"
	@echo "  3. Create a git tag"
	@echo "  4. Push to repository"