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
`5fc8a28f7a5eeb57bbc0729e753e89139f323920b4dea4c4a3e24a3c4c677fbc`).

Compile with the portable TinyTeX installation:

```powershell
$texbin = 'D:\AAAI2027_GNN_HARP\tools\TinyTeX\portable\TinyTeX\bin\windows'
& "$texbin\latexmk.exe" -pdf -interaction=nonstopmode -halt-on-error main.tex
```
