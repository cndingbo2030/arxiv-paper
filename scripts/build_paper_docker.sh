#!/usr/bin/env bash
# Build the paper PDF using TeX Live inside Docker.
#
# Usage:
#   scripts/build_paper_docker.sh
#
# Optional: override image (default matches a full TeX Live with latexmk):
#   TEXLIVE_IMAGE=ghcr.io/xu-cheng/texlive-full:latest scripts/build_paper_docker.sh
#
# PDFs land in paper/build/main.pdf (per paper/.latexmkrc $out_dir and -jobname=main).
#
# Native install (optional; BasicTeX pkg needs your password in Terminal):
#   brew install --cask basictex && eval "$(/usr/libexec/path_helper)"
#   brew install latexmk
#   sudo "$(dirname "$(which tlmgr)")/tlmgr" update --self
#   scripts/build_paper.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPER_DIR="${PAPER_DIR:-$SCRIPT_DIR/../paper}"
IMAGE="${TEXLIVE_IMAGE:-danteev/texlive:latest}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running or not reachable. Start Docker Desktop and retry." >&2
  exit 1
fi

mkdir -p "$PAPER_DIR/build"

docker run --rm -v "$PAPER_DIR:/work" -w /work "$IMAGE" \
  sh -c 'latexmk -C main.tex 2>/dev/null || true; exec latexmk -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex'

echo "Output: $PAPER_DIR/build/main.pdf"
