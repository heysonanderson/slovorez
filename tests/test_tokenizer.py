import pytest

from slovorez.core.tokenizer import FFTokenizer


def test_fftokenizer_rejects_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Not a file"):
        FFTokenizer(tmp_path)


def test_fftokenizer_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Not a file"):
        FFTokenizer(tmp_path / "missing.txt")
