import numpy as np
import pytest

from da_edgeformer.baselines.external import load_external_predictions


def test_external_prediction_contract_checks_sparse_label_mask(tmp_path) -> None:
    path = tmp_path / "predictions.npz"
    labels = np.array([True, False, False])
    np.savez(
        path,
        power=np.zeros((3, 2), dtype=np.float32),
        state=np.zeros((3, 2), dtype=bool),
        appliances=np.array(["a", "b"]),
        visible_labels=labels,
    )
    power, state = load_external_predictions(path, 3, ["a", "b"], labels)
    assert power.shape == state.shape == (3, 2)
    with pytest.raises(AssertionError):
        load_external_predictions(path, 3, ["a", "b"], ~labels)
