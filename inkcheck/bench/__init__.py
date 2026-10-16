"""The evaluation harness: which files, run under which conditions, scored how.

    python -I -m inkcheck.bench.manifest            # data/manifest.jsonl from what is on this machine
    python -I -m inkcheck.bench.run --name main     # runs/main/results.jsonl (resumable)
    python -I -m inkcheck.bench.metrics runs/main   # the tables in docs/results.md

Real handwriting and anything made from it stay under data/ and runs/
(git-ignored); only aggregate numbers go into docs/results.md.

The harness runs many worker processes; each is held to one thread here,
before numpy or OpenCV load, or their own thread pools multiply into
hundreds of threads fighting over the cores.
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import cv2 as _cv2  # noqa: E402

_cv2.setNumThreads(0)  # 0 = run sequentially (1 is reported as the core count)
