import numpy as np
import pytest

from slovorez.core.codec import _viterbi, _viterbi_batch


def test_viterbi_batch_matches_single_word_decoder_for_variable_lengths():
    rng = np.random.default_rng(42)
    lengths = np.array([1, 2, 6, 3, 6], dtype=np.int32)
    logp = rng.normal(size=(len(lengths), 6, 4)).astype(np.float32)

    # Padding must not affect either the final state or the backtrace.
    for i, length in enumerate(lengths):
        logp[i, length:] = 1e6

    trans = np.zeros((4, 4), dtype=np.float32)
    trans[0, 0] = -1e9
    trans[3, 1] = -1e9
    start = np.array([0.0, 0.0, -1e9, 0.0], dtype=np.float32)
    end = np.array([-1e9, 0.0, 0.0, 0.0], dtype=np.float32)

    actual = _viterbi_batch(logp, trans, start, end, lengths)

    for i, length in enumerate(lengths):
        expected = _viterbi(logp[i, :length], trans, start, end)
        np.testing.assert_array_equal(actual[i, :length], expected)
        np.testing.assert_array_equal(actual[i, length:], 0)


@pytest.mark.parametrize("lengths", [[0, 2], [1, 7]])
def test_viterbi_batch_rejects_invalid_lengths(lengths):
    logp = np.zeros((2, 6, 4), dtype=np.float32)
    trans = np.zeros((4, 4), dtype=np.float32)
    start = np.zeros(4, dtype=np.float32)
    end = np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="lengths"):
        _viterbi_batch(logp, trans, start, end, np.array(lengths))
