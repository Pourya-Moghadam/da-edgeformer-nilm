from pathlib import Path

import pytest

from da_edgeformer.config import ExperimentConfig, load_config


def test_manuscript_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "manuscript.yaml"
    config = load_config(path)
    assert config.model.window_size == 256
    assert config.controller.minimum_new_labels == 32


def test_config_rejects_insufficient_purge() -> None:
    config = ExperimentConfig()
    config.data.purge_samples = 10
    with pytest.raises(ValueError, match="purge_samples"):
        config.validate()
