#!/usr/bin/env bash
# Build the medRxiv preprint PDFs from submission/manuscript.md and submission/supplementary.md.
# The corresponding author's postal address and phone number are dropped (a preprint is public), as are
# the word-count, running-head and journal alt-text lines. Rerun after any manuscript change:
#   bash scripts/build_preprint_pdf.sh [output_dir]      (default: submission/medrxiv)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-submission/medrxiv}
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# Captions already carry "Figure N." in the text, so drop LaTeX's own label; wrap long prompt lines and file paths.
cat > "$TMP/header.tex" <<'TEX'
\usepackage{caption}
\captionsetup[figure]{labelformat=empty}
\usepackage{fvextra}
\let\verbatim\relax
\let\endverbatim\relax
\DefineVerbatimEnvironment{verbatim}{Verbatim}{breaklines,breakanywhere,fontsize=\small}
\let\oldus\_
\renewcommand{\_}{\oldus\allowbreak}
TEX

build() {  # build <source.md> <output.pdf>
  local title
  title=$(sed -n '1s/^# //p' "$1")
  sed -E -e '1d' -e '/^\*(Address|Phone|Word count|Alt text):/d' -e '/^\*\*Running head:\*\*/d' \
      -e 's/^## (\*\*(Abstract|Key Points|Plain Language Summary|Lay Summary)\*\*)/# \1/' "$1" > "$TMP/src.md"
  pandoc "$TMP/src.md" -o "$2" --pdf-engine=xelatex --resource-path=submission -H "$TMP/header.tex" \
    -M title="$title" -M date="$(date +%-d/%-m/%Y)" \
    -V mainfont="Times New Roman" -V fontsize=11pt -V geometry:margin=2.5cm \
    -V linestretch=1.15 -V colorlinks=true -V urlcolor=blue -V linkcolor=black
}

build submission/manuscript.md "$OUT/manuscript_medrxiv.pdf"
build submission/supplementary.md "$OUT/supplementary_medrxiv.pdf"
ls -l "$OUT"/*.pdf
