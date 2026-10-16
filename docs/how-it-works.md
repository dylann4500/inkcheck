# How inkcheck works

inkcheck reads a homework PDF or page image and looks for handwriting that
software assembled from samples: the way Scribe writes, placing crops of a
writer's own letters (from a filled practice sheet) with a small random turn,
slant and stretch on each. It reports evidence a person can check by eye.

It runs four checks ("layers") over one shared reading of the ink, then a
verdict step combines them.

```
file ──load──▶ pages (pixels, dpi, how each page is stored)
                 │
                 ├─▶ provenance   what the file says about how it was made
                 ├─▶ ink ─▶ pieces ─▶ reuse   the same letter written twice
                 └─▶ ink ─▶ edge patches ─▶ texture   what the ink's edges look like
                                   │
                              verdict ─▶ JSON + HTML evidence report
```

## Loading (`load.py`)

A PDF page that is one image covering the page (how Scribe, scanners and
"flatten" exports store pages) gives that image's own pixels, untouched.
Anything else is rendered at 288 dpi, Scribe's export resolution. Each page
also records its facts: images and their codecs, vector ink segments,
visible text (printed text the PDF holds, with boxes) and invisible text (the
handwriting-recognition layer note apps add). Image files load with their
EXIF orientation and camera details.

## Layer 1: provenance (`provenance.py`)

- **Who made the file**: the PDF producer or EXIF software is matched against
  a table (note apps, Apple devices, scanner apps, copiers, Print to PDF,
  PDF libraries).
- **How pages are stored**: vector ink (a note app), a single image per page,
  JPEG or lossless.
- **Where the pixels came from**: a camera or scanner leaves sensor noise and
  uneven light; software leaves exactly uniform paper (one grey value for most
  of the page) and no noise.
- **Known generator layout**: Scribe's Export PDF is pdf-lib, one lossless
  image per page at exactly 2× or 4× a 612×792 page. Matching it is the
  strongest thing this layer says.

All of it is context. Metadata is easy to rewrite, a screenshot is a
rendered raster too, and printing and scanning erase this layer. It can make
a page **suspicious**, never **generated**.

## Ink and pieces (`ink.py`)

1. **Darkness against the page's own paper.** A coarse median of the page
   estimates the paper under every pixel, so uneven light and coloured
   paper drop out.
2. **Ruled and grid lines out.** A long thin opening finds them and is
   subtracted (a top-hat), so a letter that crosses a line keeps its ink.
3. **Hysteresis threshold, then connected pieces.** Specks are dropped, and
   so is anything taller than six letter heights (figures, big brackets).
4. **Pen width** is ink area over centreline length, with anti-aliased edge
   pixels counted by their share of ink. Within 5% on drawn strokes; it sets
   every scale that follows.

## Layer 2: reuse (`shapes.py`, `register.py`, `reuse.py`)

**Why it works.** A person never writes a letter the same way twice. A
generator built from a few samples per character does, with a small affine
nudge. Undo the nudge and the two copies fall on top of each other.

1. **Complex pieces only.** A piece's *complexity* is its centreline length
   in pen widths, discounted to nothing for straight strokes (tortuosity
   under 1.3). Dots, dashes, `1`, `l` and slashes repeat closely in anyone's
   hand. On the first real pages tested, every near-twin below the bar was such a stroke;
   above the bar (16 pen widths) real near-twins fell to 0.08%.
2. **Candidate twins.** Each piece's darkness, stretched to 24×24 and
   blurred, is a descriptor. The nearest five, with similar size and aspect,
   are candidates, searched across all pages of the document.
3. **Alignment.** ECC (`cv2.findTransformECC`, affine) runs coarse to fine
   on blurred darkness. Thin strokes (2× exports, small photos) are enlarged
   to about 3 px first; very large pieces are shrunk. Pairs whose coarse
   correlation stays below 0.97 stop there.
4. **The measure.** RMS stroke distance after alignment, in pen widths,
   read from the leftover darkness against the darkness gradient
   (b ≈ a + ∇a·u). It does not depend on resolution. Copies of one bitmap
   sit 0.01–0.06 apart; a letter written twice by a person, 0.15–0.4.
   A **twin** is under 0.08.
5. **Sorting the twins.** Three things repeat exactly besides reused
   samples, and all three are taken out:
   - **Printed type.** On a printed or scanned worksheet, every copy of a
     letter in one font is identical. But so are the letters a library hand
     copies whole from its sentence pages (Scribe places them "never
     jittered"), so how twins repeat cannot decide alone. A group of twins is
     set aside as type only when the printed-type model (`printed.py`)
     reads the median piece of the group as type.
     - The model is a small network over each piece's HOG and named stroke
       numbers. It was trained on 196 faces from 64 ordinary text font
       families, including Computer Modern (LaTeX), DejaVu and STIX;
       handwriting-style fonts were left out, because text set in one is
       generated handwriting.
     - With whole families and hands held out, it recognises 96.7% of
       printed letters and calls 5.4% of handwritten letters type. It only
       ever judges a group of twins, by its median piece, so a few
       handwritten letters read as type do not excuse a generator.
     - On a flat page, two copies differ by size at most (lines may be set
       at different sizes), never by a turn or slant.
     - In a photo, perspective adds a gentle, smooth change across the page.
     - So when one smooth field (a gradient, plus a constant per page pair
       for separate scans) explains most of a page's twins (RANSAC), those
       twins are type.
     - A generator's twins each carry their own random turn and slant, and
       no field explains them.
     - A line whose twins nearly all differ by size alone is type too, and a
       line that is mostly type is type throughout.
     - Scanning breaks printed letters into fragments (the bowl of a `g`
       comes away from its tail) that the model cannot read on their own.
       A twin that moved rigidly (within 3.5% and 1°), with three or more
       printed pieces within three letter heights, is type as well. This
       rule was written after four CVL forms were called generated for
       exactly this reason, so CVL's false-alarm rate after it is not an
       independent estimate.
   - **Repeated printed words.** Three or more twins moved by one shared map
     whose letters are type, as above.
   - **Copied blocks.** Three or more twins moved by one shared map whose
     letters are not type: a writer's own copy and paste (one set of real
     notes in the benchmark has a lasso-copied matrix). Reported, never counted.
6. **The test.** Twins come in pairs, so counting twinned pieces would count
   one chance coincidence twice. The statistic is the number of **extra
   copies**: group the reused twins into clusters and add up (size − 1).
   With n complex pieces and the chance rate r in `calibration.json`
   (2 per 1,000, fixed before the CVL and MathWriting pages were seen), the
   binomial tail P(X ≥ e) is the chance of that many copies on real
   handwriting. **Generated** needs e ≥ 4 and a chance below 1e-6;
   **suspicious** needs a chance below 0.004. The rate measured on all the
   real pages is lower than r; see the paper's operating-point section for
   why the pre-set rate is kept.
7. **Too little.** Under 15 complex pieces, or a pen under 0.8 px wide: the
   layer says so instead of guessing.

## Layer 4: texture (`texture.py`)

Edge patches (32 px, the page rescaled so the pen is 4 px wide) are described
by numbers a person can name:

- the spread of darkness on the edge (a renderer's anti-aliasing comes in
  steps);
- edge sharpness;
- outline detail by frequency band (Scribe's letters are 144-dpi crops
  enlarged and re-thresholded);
- how solid the stroke's core is.

A logistic model scores each patch. It learns whatever separates its
training pages, including one renderer from another: with a source held out
of training it called 99% of CVL's pages and all the tablet pages generated
(`results.md`). So it can only make a page **suspicious**, and no texture
model is installed by default.

## Verdict (`verdict.py`)

| Label | When |
|---|---|
| generated | the reuse test is significant |
| suspicious | Scribe's export layout, a weak reuse result (chance below 0.004), or a strong texture score. A merely clean, rendered-looking image is not suspicious: scanners that whiten the paper make those too |
| no-evidence | none of the above |
| too-little | the reuse test could not run and nothing else is suspicious |
| error | the file could not be read |

It never says a page was written by hand.

## The trained baseline (`bench/learned.py`)

For comparison, a convolutional network (seven 3×3 convolutions, about 0.6
million weights) learns "generated" against "real" from 64×64 patches of the
same normalised ink. It is trained and tested under several protocols:
a random split, each real source held out, each generator held out, and
undegraded training only. Within its training distribution it beats the
reuse test. With a source held out it calls most of that source's real
pages generated, and it misses a generator it has not seen. A network at a
quarter or four times the size does the same. That is why only the
reuse test can say "generated". The numbers are in [results.md](results.md)
and the paper ([paper/inkcheck.pdf](paper/inkcheck.pdf)).

## What went wrong along the way, and what fixed it

| Problem | Found on | Fix |
|---|---|---|
| Pen width came out about 9× too big | Closed shapes: a box's inside was counted as ink | Area over centreline length |
| Old 2× exports missed (strokes ~1 px) | Energy-based residual is resolution-bound | RMS stroke distance; thin strokes enlarged before alignment |
| Short strokes twinned on real pages | Dashes, `1`s, slashes | Complexity with a straightness discount |
| Printed sentences read as reuse once the PDF text layer was gone | A writing sample, rasterized, scanned or photographed | Twins fitted to one smooth page field (type) |
| Printed lines not straight enough to call | Type's overshoot: round letters dip below the baseline | Judge print by how twins repeat, not by baseline straightness |
| Repeated printed words looked like copy-paste | The writing sample's instructions | Print is decided before blocks; a block of print is print |
| A real lasso copy called suspicious, and counted six times | Real lecture notes | Blocks counted as groups and reported only |
| Scribe library letters set aside as print (184 of 198 twins on one page) | Library hands copy sentence letters without jitter | Printed-type model; groups go by its median |
| The printed paragraph on every CVL form counted as reuse (8 of 10 sampled pages) | Computer Modern, which the model had never seen | Computer Modern (and DejaVu, STIX) in the model's training; print decided per group of twins |
| The binomial test counted both members of each twin, overstating the evidence 10²–10⁴-fold | An independent code review | The statistic counts extra copies |
| 4 of 1,604 CVL forms called generated | Scan-broken fragments of printed letters (the bowl of a `g`) | A rigid twin among printed pieces is print |
| Pure prose in an old 2× export missed | Letters 12 px tall, pen 1 px | A resolution limit: the same worksheet at 4× has 612 of 892 pieces twinned |
