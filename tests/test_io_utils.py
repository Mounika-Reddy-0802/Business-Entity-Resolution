import os
import time

from src.common.io_utils import is_fresh


def test_is_fresh_needs_newer_outputs_and_respects_force(tmp_path, monkeypatch):
    src_in, out = tmp_path / "in.tsv", tmp_path / "out.parquet"
    src_in.write_text("x")
    assert not is_fresh([out], [src_in])                 # output missing
    future = time.time() + 3600
    out.write_text("y")
    os.utime(out, (future, future))
    assert is_fresh([out], [src_in])
    monkeypatch.setenv("FORCE", "1")
    assert not is_fresh([out], [src_in])
    monkeypatch.delenv("FORCE")
    os.utime(src_in, (future + 10, future + 10))          # input changed after the output
    assert not is_fresh([out], [src_in])
