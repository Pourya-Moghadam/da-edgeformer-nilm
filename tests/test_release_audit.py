from da_edgeformer.release_audit import release_audit


def test_release_audit_detects_unfinished_marker_and_ignores_artifacts(tmp_path) -> None:
    (tmp_path / "clean.py").write_text("value = 1\n", encoding="utf-8")
    ignored = tmp_path / "outputs"
    ignored.mkdir()
    (ignored / "private.npz").write_bytes(b"data")
    assert release_audit(tmp_path)["ready"]
    (tmp_path / "bad.md").write_text("FIX" + "ME before release\n", encoding="utf-8")
    report = release_audit(tmp_path)
    assert not report["ready"]
    assert "unfinished marker: bad.md" in report["failures"]
