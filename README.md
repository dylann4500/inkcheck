# inkcheck

Finds handwriting that software assembled from samples, the way
[Scribe](https://scribe-homework.site) writes, and shows the evidence a
person can check by eye.

A person never writes a letter the same way twice. Software that builds
handwriting from a few samples per character does, nudged each time by a
small turn, slant or stretch. inkcheck undoes the nudge and finds the copies.
That evidence is in the pixels, so it survives exporting to an image-only
PDF, JPEG compression, printing and scanning, and phone photos.

<!-- RESULTS -->
## Results

- **The glyph-reuse test** called 97.1% of Scribe's document-conditions generated (99.7% at today's 4× export resolution, 86.1% of the older 2× exports), across five conditions including print-and-scan and phone photos.
- **False alarms on real handwriting:** 0 of 3,609 document-conditions (95% upper bound 0.08%); 7 more were marked suspicious.
- **MathWriting's generator** (a pool of 6,423 glyphs) is the hard case: 25.3% caught as exported.
- **The trained CNN** wins inside its training distribution (AUC 0.975 against the test's 0.939) but called 90.9% of CVL's pages generated when CVL was held out of training, 94.5% of the tablet pages, and 100.0% of MathWriting's human pages; it caught 2.8% of Scribe's documents when Scribe was held out.
- **Capacity sweep:** at 0.15 M, 0.58 M and 2.33 M weights the CNN still called 92.5%, 91.5% and 90.3% of held-out CVL generated.
- **Same writer:** 4,000 single MathWriting expressions held 18,805 complex letters and 0 extra copies (95% upper bound 0.16 per 1,000).
- **Closest look-alikes:** 67.8% of Scribe's complex letters have a twin under 0.08 pen widths; 0.0% of real letters do.
- **What the real copies are:** of the 49 groups of chance copies on CVL (all 43 pages that had any), 48 were printed type the print rules missed (43 scan-broken bowls of a printed g, five printed 7s) and one was handwriting (two cursive l's at d = 0.072).

All numbers: [docs/results.md](docs/results.md). The paper: [docs/paper/inkcheck.pdf](docs/paper/inkcheck.pdf).
<!-- /RESULTS -->

## Paper

**Copies Give It Away: Detecting Sample-Assembled Handwriting by Glyph Reuse, Compared with a Trained Classifier**
([PDF](docs/paper/inkcheck.pdf), LaTeX source in [docs/paper](docs/paper)). Every number in it is
written by `docs/paper/make_figures.py` from the benchmark's result files.

```bibtex
@misc{nguyen2026copies,
  author       = {Nguyen, Dylan},
  title        = {Copies Give It Away: Detecting Sample-Assembled Handwriting by Glyph Reuse,
                  Compared with a Trained Classifier},
  year         = {2026},
  howpublished = {\url{https://github.com/dylann4500/inkcheck}}
}
```

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

Python 3.11 or newer. Dependencies: numpy, scipy, OpenCV (headless),
scikit-image, PyMuPDF, Pillow, scikit-learn.

## Use

```bash
inkcheck scan homework.pdf                         # a verdict and the reasons
inkcheck scan *.pdf photo.jpg --html reports/      # plus one evidence report per file
inkcheck scan homework.pdf --json result.json      # machine-readable
inkcheck serve                                     # a page on http://127.0.0.1:8765 to drop files on
```

| Verdict | Meaning | Exit code |
|---|---|---:|
| **generated** | Many complex letters are copies of one another, far more than real handwriting ever shows | 2 |
| **suspicious** | Worth a closer look: weaker reuse evidence, a known generator's export layout, or a confident texture model | 1 |
| **no evidence found** | Nothing above was found. Not proof of anything | 0 |
| **too little to judge** | Too few complex letters, or too low a resolution | 3 |
| **could not read** | The file is broken or password-protected | 3 |

inkcheck never says a page was written by hand, and a verdict is evidence for
a person to weigh, never a judgment about a student.

### Reading the report

The HTML report is one self-contained page with these sections:

- **Verdict** and its reasons.
- **Letters that are copies of one another:** one row per repeated shape, with
  every copy cut from the page. On the right, the closest two are laid over
  each other after the turn, slant and stretch between them are undone: red
  is one, blue the other, black where they agree. Two copies of one sample
  come out almost all black. Two letters written by a person never do.
- **Blocks of ink that appear twice** (copy and paste) and **printed text**,
  both left out of the test.
- **What each check found**, and the file's **facts**.
- **Pages** with every repeated letter boxed, one colour per shape.

The report embeds the page images, so it is written only where you ask.

## How it works

Four checks over one reading of the ink:

- **Provenance:** which program made the file and how its pages are stored.
  Scribe's Export PDF has an exact layout.
- **Reuse:** complex letters aligned with their look-alikes. Twins closer than
  0.08 pen widths are counted, after printed type and copy-paste are
  recognised and set aside. The count is tested against a chance rate for real
  handwriting.
- **Texture:** a small model over named features of the ink's edges.
- **Verdict:** only reuse can say "generated".

The full write-up, including the false alarms found along the way and how
each was fixed: [docs/how-it-works.md](docs/how-it-works.md). Measured
results: [docs/results.md](docs/results.md).

## Limits

- **A generator built differently.** One that draws every letter afresh
  (a handwriting-synthesis model rather than sample reuse), or adds a
  different non-rigid distortion to every copy, leaves no twins. The reuse
  test would find nothing, and only the weaker checks would remain.
- **Short pages.** A page with fewer than about 15 complex letters (one short
  answer) cannot be judged.
- **Very low resolution.** Strokes under about 0.8 px wide (an old 2× export
  shrunk again) cannot be compared. Pure prose in a library hand is the
  hardest case. In one of Scribe's old 2× exports at its smallest ruling, the
  letters were 12 px tall and no twins were found. The same worksheet as
  today's 4× export had 612 of 892 pieces twinned.
- **A handwriting font with no jitter at all** looks like printed type and is
  set aside with it. Scribe always jitters, even at its neatest setting.
- **The false-alarm rate rests on the real handwriting tested:** the
  author's pages, two other writers' practice sheets, the CVL database's 310 writers
  and MathWriting's contributors (see results). Every one of those sources
  was used while building the detector, so the rate shows that the known
  failures are fixed, not how often unknown ones occur. Very consistent
  writers, or very small writing, could twin more often.
- **A large sample pool.** MathWriting's own generator draws from 6,423
  glyphs, so a page repeats few of them; the test catches about a fifth of
  its pages. Scribe's one or two samples per character are the easy case.
- **Printed text needs the printed-type model.** Printed worksheets repeat
  their letters exactly. inkcheck sets them aside with a shape model trained
  locally (`inkcheck.bench.train_printed`) on 64 font families, including
  Computer Modern for LaTeX worksheets. A fresh clone has no model and falls
  back to geometry alone, which handles flat print but not every scan.

## Data and privacy

Real handwriting, and everything made from it (including Scribe pages in a
real person's hand), lives in `data/` and `runs/`. Git ignores both, and
nothing from them is ever committed. Committed figures and tables hold
aggregate numbers only.

- **Scribe pages** for the benchmark were rendered by Scribe's own renderer,
  in headless Chrome and identical to the app, by a generator that drives
  Scribe's private code. That generator is not part of this repository, and
  neither are the pages or the hands they were written in.
- **The CVL database** (Kleber et al., Zenodo record 1492267) provides the
  many-writer real handwriting. It is CC BY-NC 4.0, for non-commercial
  research only. It is used here to measure the detector, and must not be
  used to tune a commercial product.
- **MathWriting** (Gervais, Fadeeva and Maksai, 2024; CC BY-NC-SA 4.0)
  provides touchscreen handwriting and a second, independent generator
  (its synthetic, glyph-stitched expressions). Same terms: research only.

## Reproducing the benchmark

The public datasets go under `data/public/` (git-ignored):

- **CVL:** `cvl-database-1-1.zip` from Zenodo record 1492267, unpacked into `data/public/cvl/`.
- **MathWriting:** `mathwriting-2024.tgz` from Google, unpacked into `data/public/mathwriting/`; then
  `python -I -m inkcheck.bench.mathwriting --source data/public/mathwriting/mathwriting-2024 --pages 150`
  builds the worksheet-like pages.
- **Your own files:** Scribe Export PDFs and real tablet notes are read from `INKCHECK_LOCAL`
  (default `~/Downloads`). Real notes are listed, with who wrote them, in `data/real_files.json`
  (`{"file name.pdf": "writer id"}`), which stays on your machine.

The Scribe pages rendered for the paper cannot be rebuilt from this repository (see above).

## Development

```bash
.venv/bin/pytest                                   # unit tests (procedurally drawn pages only)
.venv/bin/python -I -m inkcheck.bench.manifest     # data/manifest.jsonl from what is on this machine
.venv/bin/python -I -m inkcheck.bench.run --name main
.venv/bin/python -I -m inkcheck.bench.metrics runs/main [--calibrate]
.venv/bin/python -I -m inkcheck.bench.train_texture [--save]
.venv/bin/python -I -m inkcheck.bench.learned extract    # CNN patches (data/ml)
.venv/bin/python -I -m inkcheck.bench.learned run [--width 1]   # train and test the CNN per protocol (runs/learned)
.venv/bin/python -I -m inkcheck.bench.compare --twin runs/final # the CNN against the twin test
.venv/bin/python -I -m inkcheck.bench.mw_null --source data/public/mathwriting/mathwriting-2024   # same-writer repeats
.venv/bin/python -I -m inkcheck.bench.distances          # closest look-alike distances (paper figure)
.venv/bin/python -I docs/paper/make_figures.py           # paper numbers, tables, figures; docs/results.md; README results
cd docs/paper && tectonic inkcheck.tex                   # the paper
```

```
inkcheck/
  load.py         PDF or image → pages, with how each page is stored
  provenance.py   layer 1
  ink.py          ink, ruling removal, pieces, pen width
  shapes.py       complexity and descriptors
  register.py     ECC alignment, stroke distance
  reuse.py        layer 2: twins, printed type, copied blocks, the test
  texture.py      layer 4: edge-patch model
  verdict.py      the answer
  report.py       JSON and HTML evidence report
  serve.py        the local web page
  degrade.py      print-scan, photo, JPEG, half resolution, grey
  calibration.json  thresholds set from real pages
  bench/          manifest, run, metrics, train_texture, train_printed,
                  mathwriting, mw_null, distances, learned (the CNN), compare
docs/             how it works, results
docs/paper/       the paper (LaTeX, IEEE format) and the script that makes its numbers
```

## License

The code is MIT-licensed ([LICENSE](LICENSE)). The paper's evidence figure shows ink from MathWriting
(CC BY-NC-SA 4.0). No dataset is included: CVL and MathWriting keep their own non-commercial licenses.
