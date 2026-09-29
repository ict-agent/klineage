# Paper

arXiv source for *KLineage: Recovering the Missing When of Kernel Optimization by
Deoptimizing Experts*. Flat layout: every `.tex`, `.sty`, `.bst`, `.bib` and figure
sits in this directory, which is what arXiv expects.

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Layout

| File | Contents |
| --- | --- |
| `main.tex` | Preamble, title, author block, section order |
| `abstract.tex` … `conclusion.tex` | Main text, one file per section |
| `extended_experiments.tex`, `appendix_baselines.tex` | Appendices |
| `custom.bib` | Bibliography |
| `preprint.sty`, `preprint.bst` | Two-column style, no venue header, no line numbers |
| `*.pdf` | Figures, referenced by bare filename |

## Before uploading

- `main.tex` still carries a placeholder author block. Replace it.
- The abstract does not yet point at this repository.

Upload the whole directory as a tarball; arXiv compiles `main.tex` and needs no
`figures/` or `sections/` subdirectories.
