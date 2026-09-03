#!/usr/bin/env bash
# Build an arXiv-ready source tarball from paper/.
#
# arXiv requires LaTeX source, not a PDF, and rejects uploads with missing
# \input or \includegraphics targets. This collects exactly the files main.tex
# references -- nothing else -- and test-builds them in an empty directory, so
# a missing file fails here rather than on upload.
#
# Usage:  bash paper/make_arxiv_package.sh
# Output: arxiv-submission.tar.gz in the repository root.

set -euo pipefail
cd "$(dirname "$0")/.."
SRC=paper
OUT=arxiv-submission.tar.gz
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/tables" "$STAGE/figures"
cp "$SRC/main.tex" "$STAGE/"

# Resolve dependencies from main.tex rather than globbing, so unused figures
# (the repo keeps several) stay out of the upload.
grep -oE '\\input\{[^}]+\}' "$SRC/main.tex" | sed 's/.*{//;s/}//' | while read -r t; do
    cp "$SRC/$t.tex" "$STAGE/tables/"
done
grep -oE 'figures/[a-zA-Z0-9_]+\.png' "$SRC/main.tex" | sort -u | while read -r f; do
    cp "$SRC/$f" "$STAGE/figures/"
done

# The bibliography is inline (\begin{thebibliography}), so no .bib/.bbl is
# needed -- the most common cause of a failed arXiv build does not apply here.

if command -v pdflatex >/dev/null 2>&1; then
    echo "test-building in a clean directory..."
    ( cd "$STAGE" && for i in 1 2 3; do
        pdflatex -interaction=nonstopmode -halt-on-error main.tex >"build$i.log" 2>&1 \
            || { echo "BUILD FAILED - see $STAGE/build$i.log"; exit 1; }
      done
      grep -oE 'Output written on main.pdf \([0-9]+ pages' build3.log
      echo "unresolved refs/citations: $(grep -cE 'LaTeX Warning: (Reference|Citation)' build3.log)"
      echo "overfull boxes: $(grep -c Overfull build3.log)" )
    rm -f "$STAGE"/*.log "$STAGE"/*.aux "$STAGE"/*.out "$STAGE"/main.pdf
else
    echo "pdflatex not found - packaging without the build check."
fi

tar czf "$OUT" -C "$STAGE" .
echo "wrote $OUT"
tar tzf "$OUT" | grep -v '^\./$' | sed 's/^/  /'
