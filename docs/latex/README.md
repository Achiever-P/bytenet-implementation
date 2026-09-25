# LaTeX sources (institutional format)

`report.tex` and `slides.tex` follow the exact document classes/structure
required for this course: `report.tex` uses `IEEEtran` (conference style,
matching the department's internship-report template) and `slides.tex`
uses `beamer` with the Madrid theme (matching the department's standard
presentation template).

Both are self-contained and were compiled successfully with `pdflatex`
(two passes for `report.tex`, to resolve the section cross-reference);
`report.pdf` and `slides.pdf` are the compiled outputs, included here for
convenience. `figures/` holds the PNGs both documents pull from.

To recompile:
```bash
pdflatex report.tex && pdflatex report.tex   # second pass resolves refs
pdflatex slides.tex
```
Requires a TeX distribution with the `IEEEtran` class (Ubuntu/Debian:
`apt-get install texlive-publishers`).

**Before submitting:** replace `<your-username>` in both files' reference/
final slide with your actual GitHub username once the repo is pushed.
