# =====================================================================
#  Top-level Makefile -- one-stop interface for paper + experiments.
# =====================================================================

PAPER_DIR := paper
EXP_DIR   := experiments
BUILD_DIR := $(PAPER_DIR)/build
ARXIV_PKG := arxiv_submission.tar.gz

.PHONY: help paper arxiv figures experiments \
        install lint test clean clean-paper clean-all

help:
	@echo "Common targets:"
	@echo "  make paper           - build the preprint PDF"
	@echo "  make arxiv           - build a minimal arXiv source tarball"
	@echo "  make figures         - generate fresh exploratory diagnostic plots"
	@echo "  make experiments     - run smoke-test suite for experiments code"
	@echo "  make install         - install Python package in editable mode"
	@echo "  make lint            - run ruff + mypy on experiments/"
	@echo "  make test            - run pytest on experiments/"
	@echo "  make clean           - remove LaTeX build artifacts"
	@echo "  make clean-all       - clean + remove experiments results"

# ----------------- LaTeX -----------------
paper:
	cd $(PAPER_DIR) && latexmk -pdf main.tex

arxiv: paper
	bash scripts/arxiv_package.sh $(ARXIV_PKG)
	@echo "Built $(ARXIV_PKG) -- upload directly to arXiv."

# ----------------- Experiments -----------------
install:
	pip install -e $(EXP_DIR)[dev]

figures:
	bash scripts/run_all_experiments.sh figures-only

experiments:
	bash scripts/run_all_experiments.sh smoke

lint:
	cd $(EXP_DIR) && ruff check src tests scripts sweep_sparse.py
	cd $(EXP_DIR) && mypy src

test:
	cd $(EXP_DIR) && pytest

# ----------------- Cleanup -----------------
clean: clean-paper

clean-paper:
	cd $(PAPER_DIR) && latexmk -C
	rm -rf $(BUILD_DIR) $(ARXIV_PKG)

clean-all: clean
	rm -rf $(EXP_DIR)/results $(EXP_DIR)/.pytest_cache $(EXP_DIR)/.ruff_cache $(EXP_DIR)/.mypy_cache
