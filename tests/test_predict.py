import json

import pytest

from slovorez import Slovorez
from slovorez.config import SlovorezConfig

TEXT = "Мама мыла раму, а кто-то выделяется. Мама!"
WORDS = ["мама", "мыла", "раму", "а", "кто-то", "выделяется", "мама"]


@pytest.fixture(scope="module")
def config():
    return SlovorezConfig.from_pretrained("slovorez-flash")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("predict")
    model = Slovorez.from_pretrained(predictions_path=tmp / "predictions.jsonl")
    return model.predict(TEXT)


def test_bundled_model_loads_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    model = Slovorez.from_pretrained("slovorez-flash", predictions_path=tmp_path / "p.jsonl")

    assert [r["word"] for r in model.predict("кошка")] == ["кошка"]


def test_predict_keeps_word_order_and_duplicates(records):
    assert [r["word"] for r in records] == WORDS


def test_morphemes_concatenate_back_to_the_word(records):
    for r in records:
        assert "".join(m[0] for m in r["morphemes"]) == r["word"]


def test_morpheme_types_are_known_names(records, config):
    known = set(config.mapping.morpheme_types)
    for r in records:
        for text, mtype, score in r["morphemes"]:
            assert isinstance(mtype, str)
            assert mtype in known, f"{r['word']}: unknown type {mtype!r}"


def test_record_format(records, config):
    for r in records:
        assert set(r) == {"word", "morphemes", "confidence", "model"}
        assert r["model"] == config.name
        assert type(r["confidence"]) is float  # not np.float64
        assert 0.0 <= r["confidence"] <= 1.0
        for m in r["morphemes"]:
            assert isinstance(m, list) and len(m) == 3
            text, mtype, score = m
            assert isinstance(text, str) and text
            assert type(score) is float and 0.0 <= score <= 1.0


def test_predict_output_survives_json_round_trip(records):
    assert json.loads(json.dumps(records, ensure_ascii=False)) == records


@pytest.mark.parametrize(
    "text, last",
    [
        ("я видел кто-то", "кто-то"),
        ("пошёл из-за", "из-за"),
        ("бело-сине-красный", "бело-сине-красный"),
        ("летим на ту-сто", "ту-сто"),
    ],
)
def test_predict_keeps_hyphenated_word_at_the_very_end(tmp_path, text, last):
    # C16: the C++ tokenizer drops a hyphenated chain if nothing follows it.
    # predict() works around it; C4 makes the loss flaky, hence the repeats.
    model = Slovorez.from_pretrained(predictions_path=tmp_path / "p.jsonl")

    for _ in range(20):
        assert model.predict(text)[-1]["word"] == last


def test_predict_returns_the_same_records_as_process_file(tmp_path):
    # Nonsense words: none of them is in base_dict, so process_file writes all.
    words = ["бррак", "брррак", "бррррак"]
    path = tmp_path / "text.txt"
    path.write_text(" ".join(words) + "\n", encoding="utf-8")
    model = Slovorez.from_pretrained(predictions_path=tmp_path / "p.jsonl")

    model.process_file(path)
    with open(tmp_path / "p.jsonl", encoding="utf-8") as f:
        from_file = {r["word"]: r for r in map(json.loads, f)}
    from_predict = {r["word"]: r for r in model.predict(" ".join(words))}

    assert from_file.keys() == from_predict.keys()
    for word in words:
        a, b = from_file[word], from_predict[word]
        assert [m[:2] for m in a["morphemes"]] == [m[:2] for m in b["morphemes"]]
        assert [m[2] for m in a["morphemes"]] == pytest.approx([m[2] for m in b["morphemes"]])
        assert a["confidence"] == pytest.approx(b["confidence"])
        assert a["model"] == b["model"]
