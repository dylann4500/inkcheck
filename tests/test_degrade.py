import numpy as np
import pytest

import synth
from inkcheck.degrade import KINDS, degrade


@pytest.fixture(scope="module")
def page():
    return synth.human_page(seed=41, shape=(792, 612))


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_is_deterministic(page, kind):
    a, dpi_a = degrade(page, 288, kind, seed=3)
    b, dpi_b = degrade(page, 288, kind, seed=3)
    assert np.array_equal(a, b) and dpi_a == dpi_b
    assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3


def test_downscale_halves_the_resolution(page):
    out, dpi = degrade(page, 288, "downscale")
    assert dpi == 144 and out.shape[0] == page.shape[0] // 2


def test_print_scan_lands_at_300_dpi_and_adds_noise(page):
    out, dpi = degrade(page, 288, "print-scan", seed=1)
    assert dpi == 300
    assert abs(out.shape[1] - round(page.shape[1] * 300 / 288)) <= 1
    paper = out[out[..., 0] > 200]
    assert paper.std() > 1.0


def test_a_photo_changes_the_geometry_and_the_light(page):
    out, _ = degrade(page, 288, "photo", seed=2)
    corner = out[:10, :10].mean()
    assert corner < 140                     # the table shows round the page
    assert out.shape[:2] != page.shape[:2]
