# WWW manuscript

`main.tex` is the clean, anonymous WWW submission manuscript for STLA (Subgroup-Safe
Temporal Logit Adaptation). It intentionally separates:

- four-dataset, two-backbone validation-only development evidence;
- the independent hash-frozen MemeTracker one-shot test;
- a clearly labeled post-confirmation, validation-only component analysis; and
- the direct but secondary relationship to the CIKM DeDiff model.

The manuscript follows the official WWW research-track format
(`sigconf, anonymous, review`): eight self-contained main pages, with
references and an optional appendix within the twelve-page total limit.

Current compiled artifact: `D:\DeDiff\output\pdf\STLA_WWW2027_submission.pdf`
(9 pages: 8 main + 1 full references page, with no appendix; SHA-256
`44c02a21cf3f575125a6e07911ec9c179648b8dce597bb39a6df12e2a1e9155b`).

Compile with the portable TinyTeX installation:

```powershell
$texbin = 'D:\AAAI2027_GNN_HARP\tools\TinyTeX\portable\TinyTeX\bin\windows'
& "$texbin\latexmk.exe" -pdf -interaction=nonstopmode -halt-on-error main.tex
```
