import json

import pytest

from slovorez.core.cache import LogWriter


def test_flush_keeps_buffer_when_write_fails(tmp_path):
    # PY-6: a failed write must not drop the buffer; a retry writes it once.
    path = tmp_path / "predictions.jsonl"
    writer = LogWriter(path)
    record = {"word": "мужа", "morphemes": [["муж", "ROOT", 1.0], ["а", "END", 1.0]]}
    writer.write([record])

    path.mkdir()  # opening a directory for append raises OSError
    with pytest.raises(OSError):
        writer.flush()

    path.rmdir()
    writer.flush()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [record]
