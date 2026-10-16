import sys
from pathlib import Path

import pytest

# The procedural fixtures (synth.py) live beside the tests.
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def no_local_models(monkeypatch):
    """Tests run as a fresh clone does: without the models trained on this
    machine's private handwriting (inkcheck/models/, git-ignored)."""
    import inkcheck.printed as printed
    import inkcheck.texture as texture

    monkeypatch.setattr(printed, "_MODEL", None)
    monkeypatch.setattr(printed, "model_path", lambda: None)
    monkeypatch.setattr(texture, "model_path", lambda: None)
