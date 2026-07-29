#!/usr/bin/env bash
# Build the paper PDF.
#
# Requires local TeX (e.g. MacTeX / BasicTeX) and latexmk. To compile via
# Docker instead: scripts/build_paper_docker.sh
set -euo pipefail

cd "$(dirname "$0")/../paper"
latexmk -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex
