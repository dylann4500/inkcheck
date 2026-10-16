import numpy as np

import synth
import inkcheck.texture as texture
from inkcheck.ink import extract_ink
from inkcheck.load import load_document


def test_patches_sit_on_ink_edges_at_a_normalised_scale():
    ink = extract_ink(synth.generated_page(seed=61, width=6.0), 288)
    batch = texture.patches(ink, n=50, rng=np.random.default_rng(0))
    assert batch.shape == (50, texture.PATCH, texture.PATCH)
    centre = batch[:, texture.PATCH // 2 - 2 : texture.PATCH // 2 + 2, texture.PATCH // 2 - 2 : texture.PATCH // 2 + 2]
    assert (centre.max(axis=(1, 2)) > 0.2).mean() > 0.9
    rows = texture.features(batch)
    assert rows.shape == (50, texture.N_FEATURES) and np.isfinite(rows).all()


def test_without_a_model_the_layer_steps_aside(tmp_path, monkeypatch):
    monkeypatch.setattr(texture, "model_path", lambda: None)
    (tmp_path / "p.png").write_bytes(synth.png_bytes(synth.human_page(seed=62, shape=(500, 700))))
    result = texture.assess_texture(load_document(tmp_path / "p.png"))
    assert result.score is None
    assert result.findings[0].code == "no-model"


def test_a_saved_model_scores_pages(tmp_path, monkeypatch):
    path = tmp_path / "texture.npz"
    n = texture.N_FEATURES
    texture.save_model(path, np.zeros(n), np.ones(n), np.zeros(n), 0.0, "test")
    monkeypatch.setattr(texture, "model_path", lambda: path)
    (tmp_path / "p.png").write_bytes(synth.png_bytes(synth.human_page(seed=63, shape=(500, 700))))
    result = texture.assess_texture(load_document(tmp_path / "p.png"))
    assert abs(result.score - 0.5) < 1e-6
