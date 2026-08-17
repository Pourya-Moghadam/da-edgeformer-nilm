from pathlib import Path

from da_edgeformer.audit import audit_protocol
from da_edgeformer.config import load_config
from da_edgeformer.protocol import load_protocol


def test_fixed_protocol_audit_passes() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/manuscript.yaml")
    protocol = load_protocol(root / "configs/protocol.yaml", config)
    manifests = [
        root / "configs/datasets/manuscript/redd.yaml",
        root / "configs/datasets/manuscript/ukdale.yaml",
        root / "configs/datasets/manuscript/refit.yaml",
        root / "configs/datasets/manuscript/enertalk.yaml",
    ]
    report = audit_protocol(config, protocol, manifests, root / "configs/stream_orders.yaml")
    assert report["ready"], report["failures"]
    assert report["parameters"]["online_trainable"] == 73_982
