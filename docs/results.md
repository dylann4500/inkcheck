# Results

Written by `docs/paper/make_figures.py` from `runs/final` (the glyph-reuse test) and `runs/learned` (the trained CNN). Aggregate numbers only. The paper, [paper/inkcheck.pdf](paper/inkcheck.pdf), explains every number here.

## Headline

- **The glyph-reuse test** called 97.1% of Scribe's document-conditions generated (99.7% at today's 4× export resolution, 86.1% of the older 2× exports), across five conditions including print-and-scan and phone photos.
- **False alarms on real handwriting:** 0 of 3,609 document-conditions (95% upper bound 0.08%); 7 more were marked suspicious.
- **MathWriting's generator** (a pool of 6,423 glyphs) is the hard case: 25.3% caught as exported.
- **The trained CNN** wins inside its training distribution (AUC 0.975 against the test's 0.939) but called 90.9% of CVL's pages generated when CVL was held out of training, 94.5% of the tablet pages, and 100.0% of MathWriting's human pages; it caught 2.8% of Scribe's documents when Scribe was held out.
- **Capacity sweep:** at 0.15 M, 0.58 M and 2.33 M weights the CNN still called 92.5%, 91.5% and 90.3% of held-out CVL generated.
- **Same writer:** 4,000 single MathWriting expressions held 18,805 complex letters and 0 extra copies (95% upper bound 0.16 per 1,000).
- **Closest look-alikes:** 67.8% of Scribe's complex letters have a twin under 0.08 pen widths; 0.0% of real letters do.
- **What the real copies are:** of the 49 groups of chance copies on CVL (all 43 pages that had any), 48 were printed type the print rules missed (43 scan-broken bowls of a printed g, five printed 7s) and one was handwriting (two cursive l's at d = 0.072).

## Tables

Glyph-reuse test: documents called *generated*, by condition. For generated documents this is detection (documents in brackets); for real handwriting it is the false-alarm rate (95% upper bound in brackets). CVL pages, already scans, are not printed and scanned again, and their other degradations use a fixed sample of 400 pages.

| Documents | As exported | JPEG | Half resolution | Print and scan | Phone photo |
|---|---|---|---|---|---|
| Scribe (generated) | 97.9% (189) | 97.4% (189) | 91.5% (189) | 100.0% (189) | 98.9% (189) |
| MathWriting synth. (generated) | 25.3% (150) | 24.7% (150) | 19.3% (150) | 19.3% (150) | 11.3% (150) |
| Real: tablet | 0/11 [23.8%] | 0/11 [23.8%] | 0/11 [23.8%] | 0/11 [23.8%] | 0/11 [23.8%] |
| Real: CVL scans | 0/1,604 [0.2%] | 0/400 [0.7%] | 0/400 [0.7%] | – | 0/400 [0.7%] |
| Real: MathWriting human | 0/150 [2.0%] | 0/150 [2.0%] | 0/150 [2.0%] | 0/150 [2.0%] | 0/150 [2.0%] |


Sensitivity of the operating point (first row): documents called generated when one setting changes. r: chance copies per 1,000 complex letters; the measured rate is the 95% upper bound on the 584,523 complex letters of the undegraded real documents. d: the twin threshold. All conditions pooled: 3,609 real, 945 Scribe and 750 MathWriting synthetic document-conditions.

| Setting | Real | Scribe | MW synth. |
|---|---|---|---|
| r = 2.00, d < 0.08 (used) | 0 (0.0%) | 97.1% | 20.0% |
| r = 0.16 (measured) | 8 (0.22%) | 98.2% | 37.3% |
| d < 0.06 | 0 (0.0%) | 92.5% | 16.8% |
| d < 0.10 | 1 (0.03%) | 98.7% | 22.4% |


The trained CNN against the glyph-reuse test, on the documents each protocol tests (all conditions pooled). False alarms: real documents called generated. Detection: generated documents called generated. A held-out real source's row counts false alarms on that source only; a held-out generator's row counts detection of that generator only. AUC is over the same documents (–: one class only).

| Protocol | AUC Twin | AUC CNN | False alarms Twin | False alarms CNN | Detection Twin | Detection CNN |
|---|---|---|---|---|---|---|
| Random split | 0.939 | 0.975 | 0.0% | 6.9% | 59.7% | 88.9% |
| Clean-only training | 0.939 | 0.860 | 0.0% | 39.8% | 59.7% | 93.3% |
| Held-out real: CVL | 0.939 | 0.578 | 0.0% | 90.9% | 59.7% | 70.0% |
| Held-out real: tablet | 0.936 | 0.599 | 0.0% | 94.5% | 59.7% | 76.7% |
| Held-out real: MathWriting | 0.941 | 0.397 | 0.0% | 100.0% | 59.7% | 100.0% |
| Held-out generator: Scribe | 0.996 | 0.623 | 0.0% | 34.3% | 97.1% | 2.8% |
| Held-out generator: MathWriting | 0.912 | 0.487 | 0.0% | 0.4% | 20.0% | 2.1% |


Capacity sweep: the network at three widths, each trained and tested as in the CNN table. Random split: AUC. Held-out CVL: CVL documents called generated. Held-out Scribe: Scribe documents caught.

| Weights | Random split AUC | CVL held out false alarms | Scribe held out caught |
|---|---|---|---|
| 0.15 M | 0.970 | 92.5% | 4.9% |
| 0.58 M | 0.978 | 91.5% | 2.8% |
| 2.33 M | 0.982 | 90.3% | 2.0% |


Sources. Every document is also evaluated under four degradations.

| Source | Truth | Documents | Writers/hands |
|---|---|---|---|
| Scribe, real exports | generated | 13 | 1 writer |
| Scribe, rendered | generated | 176 (581 pp.) | 5 hands |
| MathWriting synthetic | generated | 150 | 6,423 glyphs |
| Tablet notes | real | 11 | 3 |
| CVL scans | real | 1,604 | 310 |
| MathWriting human | real | 150 | many (mixed) |

