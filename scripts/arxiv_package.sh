#!/usr/bin/env bash
# Package only the source files required to compile the arXiv manuscript.
set -euo pipefail

OUT="${1:-arxiv_submission.tar.gz}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -f "$ROOT/paper/build/main.bbl" ]]; then
  echo "ERROR: paper/build/main.bbl not found. Run 'make paper' first." >&2
  exit 1
fi

cp "$ROOT/paper/main.tex" "$WORK/"
cp "$ROOT/paper/macros.tex" "$WORK/"
cp "$ROOT/paper/refs.bib" "$WORK/"
cp "$ROOT/paper/build/main.bbl" "$WORK/main.bbl"
cp -r "$ROOT/paper/sections" "$WORK/"
cp -r "$ROOT/paper/appendix" "$WORK/"

tar -czf "$ROOT/$OUT" -C "$WORK" .
echo "Wrote $ROOT/$OUT"
