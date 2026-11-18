import json
import threading
import time

import pytest

import slovorez.slovorez as slovorez_module
from slovorez import Slovorez

@pytest.fixture
def model(tmp_path):
    return Slovorez.from_pretrained(predictions_path=tmp_path / "predictions.jsonl")


@pytest.fixture
def small_text(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("фронтенд девопс около-нулевой\n", encoding="utf-8")
    return path

def test_threaded_mode_raises_when_worker_fails(model, small_text, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(model._model, "predict", broken_predict)

    with pytest.raises(RuntimeError, match="boom"):
        model.process_file(small_text, threaded_mode=True)


@pytest.fixture
def many_unseen_words_text(tmp_path):
    # 60 distinct nonsense words, none of them in base_dict: "брак", "браак", ...
    words = ["бр" + "а" * i + "к" for i in range(1, 61)]
    path = tmp_path / "many.txt"
    path.write_text(" ".join(words) + "\n", encoding="utf-8")
    return path


def test_threaded_mode_does_not_hang_when_all_workers_fail(
        model, many_unseen_words_text, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(model._model, "predict", broken_predict)

    # One worker dies on its first chunk; with model_batch=1 every word is a
    # chunk, so 60 words overflow chunk_queue (16) and the main thread blocks.
    raised = []

    def run():
        try:
            model.process_file(
                many_unseen_words_text,
                threaded_mode=True,
                max_workers=1,
                model_batch=1,
            )
        except Exception as e:
            raised.append(e)

    # Run in a daemon thread so a hang fails the test instead of freezing pytest.
    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=10)

    assert not runner.is_alive(), "process_file hung after the worker failed"
    assert len(raised) == 1
    assert isinstance(raised[0], RuntimeError)
    assert "boom" in str(raised[0])


def test_threaded_mode_does_not_hang_when_one_of_two_workers_fails(
        model, many_unseen_words_text, monkeypatch):
    real_predict = model._model.predict
    calls = 0
    calls_lock = threading.Lock()

    # Only the very first call fails: one worker dies, the other stays alive.
    def predict_failing_once(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            raise RuntimeError("boom")
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(model._model, "predict", predict_failing_once)

    # After stop is set the main thread no longer puts None into chunk_queue,
    # so the surviving worker must notice stop by itself instead of waiting
    # in chunk_queue.get() forever.
    raised = []

    def run():
        try:
            model.process_file(
                many_unseen_words_text,
                threaded_mode=True,
                max_workers=2,
                model_batch=1,
            )
        except Exception as e:
            raised.append(e)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=10)

    assert not runner.is_alive(), "process_file hung: a surviving worker did not stop"
    assert len(raised) == 1
    assert isinstance(raised[0], RuntimeError)
    assert "boom" in str(raised[0])


def test_threaded_mode_does_not_hang_when_idle_worker_waits_for_work(
        model, monkeypatch):
    # The usual production case: the main thread (tokenizer + dedup) is slower
    # than the model, so chunk_queue is empty and workers wait in get().
    # Timeline:
    #   1. the tokenizer gives 2 words -> 2 chunks (model_batch=1);
    #   2. worker A takes chunk 1, waits 0.5 s and fails;
    #   3. worker B finishes chunk 2 and waits in chunk_queue.get() on an empty queue;
    #   4. the tokenizer "thinks" until A has failed, then gives 2 more words;
    #   5. the main thread sees stop and puts no None -> B must notice stop itself.
    worker_failed = threading.Event()

    class SlowTokenizer:
        def __init__(self, file_path):
            self.calls = 0

        def set_batch_size(self, n): pass
        def set_filter(self, token_type): pass
        def set_token_min_len(self, n): pass
        def set_token_max_len(self, n): pass

        def get_batch_tokens(self):
            self.calls += 1
            if self.calls == 1:
                return ["бррак", "брррак"]
            if self.calls == 2:
                worker_failed.wait(timeout=5)
                time.sleep(0.3)   # let the failing worker set stop
                return ["бррррак", "брррррак"]
            return []

    monkeypatch.setattr(slovorez_module, "FFTokenizer", SlowTokenizer)

    real_predict = model._model.predict
    calls = 0
    calls_lock = threading.Lock()

    def predict_failing_once_slowly(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            time.sleep(0.5)   # the other worker goes idle meanwhile
            worker_failed.set()
            raise RuntimeError("boom")
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(model._model, "predict", predict_failing_once_slowly)

    raised = []

    def run():
        try:
            # The path is ignored: SlowTokenizer does not read files.
            model.process_file("unused.txt", threaded_mode=True, max_workers=2, model_batch=1)
        except Exception as e:
            raised.append(e)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=10)

    assert not runner.is_alive(), "process_file hung: an idle worker stayed in get()"
    assert len(raised) == 1
    assert isinstance(raised[0], RuntimeError)
    assert "boom" in str(raised[0])


def _run_in_thread(target, timeout=10):
    """Run target in a daemon thread; return (finished, raised exceptions)."""
    raised = []

    def run():
        try:
            target()
        except Exception as e:
            raised.append(e)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=timeout)
    return not runner.is_alive(), raised


def _written_words(model):
    path = model._writer.path
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["word"] for line in f if line.strip()]


def test_threaded_mode_raises_when_writer_fails(model, small_text, monkeypatch):
    def broken_write(results):
        raise OSError("disk full")

    monkeypatch.setattr(model._writer, "write", broken_write)

    finished, raised = _run_in_thread(
        lambda: model.process_file(small_text, threaded_mode=True, max_workers=1)
    )

    assert finished, "process_file hung after the writer failed"
    assert len(raised) == 1
    assert isinstance(raised[0], OSError)
    assert "disk full" in str(raised[0])


def test_threaded_mode_writes_everything_decoded_before_worker_failed(
        model, many_unseen_words_text, monkeypatch):
    # Every chunk that got decoded is put into writer_queue. The writer must
    # drain the queue up to None, not stop as soon as `stop` is set.
    decoded = []
    real_decode = model._tokenizer.decode_predictions_batch

    def recording_decode(*args, **kwargs):
        results = real_decode(*args, **kwargs)
        decoded.extend(r["word"] for r in results)
        return results

    monkeypatch.setattr(model._tokenizer, "decode_predictions_batch", recording_decode)

    real_predict = model._model.predict
    calls = 0
    calls_lock = threading.Lock()

    def predict_failing_on_tenth_call(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 10:
            raise RuntimeError("boom")
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(model._model, "predict", predict_failing_on_tenth_call)

    # A slow writer lets results pile up in writer_queue before stop is set.
    real_write = model._writer.write

    def slow_write(results):
        time.sleep(0.05)
        real_write(results)

    monkeypatch.setattr(model._writer, "write", slow_write)

    finished, raised = _run_in_thread(
        lambda: model.process_file(
            many_unseen_words_text,
            threaded_mode=True,
            max_workers=2,
            model_batch=1,
        )
    )

    assert finished, "process_file hung"
    assert len(raised) == 1
    assert isinstance(raised[0], RuntimeError)
    assert decoded, "nothing was decoded before the failure; the test checks nothing"
    missing = set(decoded) - set(_written_words(model))
    assert not missing, f"decoded but not written: {sorted(missing)}"


@pytest.mark.parametrize("exc", [RuntimeError("tokenizer broke"), KeyboardInterrupt()])
def test_threaded_mode_stops_all_threads_when_main_thread_fails(model, monkeypatch, exc):
    # The main thread fails mid-file (a tokenizer error or Ctrl+C). Workers and
    # the writer must stop, what was computed must be written, and the original
    # exception must come out. KeyboardInterrupt is not an Exception, so a plain
    # `except Exception` would miss it.
    first_batch = ["бррак", "брррак", "бррррак"]

    class FailingTokenizer:
        def __init__(self, file_path):
            self.calls = 0

        def set_batch_size(self, n): pass
        def set_filter(self, token_type): pass
        def set_token_min_len(self, n): pass
        def set_token_max_len(self, n): pass

        def get_batch_tokens(self):
            self.calls += 1
            if self.calls == 1:
                return first_batch
            time.sleep(0.3)   # let workers pick up the first batch
            raise exc

    monkeypatch.setattr(slovorez_module, "FFTokenizer", FailingTokenizer)

    threads_before = set(threading.enumerate())
    raised = []

    def run():
        try:
            model.process_file("unused.txt", threaded_mode=True, max_workers=2, model_batch=1)
        except BaseException as e:
            raised.append(e)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=10)

    assert not runner.is_alive(), "process_file hung after the main thread failed"
    assert raised == [exc], f"expected the original exception, got {raised!r}"
    leftover = set(threading.enumerate()) - threads_before
    assert not leftover, f"pipeline threads still alive: {leftover}"
    assert sorted(_written_words(model)) == sorted(first_batch)


@pytest.fixture
def unseen_words_text(tmp_path):
    # Nonsense words, none of them in base_dict, fewer than model_batch.
    words = ["бррак", "брррак", "бррррак"]
    path = tmp_path / "unseen.txt"
    path.write_text(" ".join(words) + "\n", encoding="utf-8")
    return path, words


def test_sequential_mode_writes_the_tail_shorter_than_model_batch(model, unseen_words_text):
    path, words = unseen_words_text

    model.process_file(path, threaded_mode=False)

    assert sorted(_written_words(model)) == sorted(words)


@pytest.mark.parametrize("exc", [RuntimeError("tokenizer broke"), KeyboardInterrupt()])
def test_sequential_mode_flushes_computed_results_when_tokenizer_fails(model, monkeypatch, exc):
    first_batch = ["бррак", "брррак", "бррррак"]

    class FailingTokenizer:
        def __init__(self, file_path):
            self.calls = 0

        def set_batch_size(self, n): pass
        def set_filter(self, token_type): pass
        def set_token_min_len(self, n): pass
        def set_token_max_len(self, n): pass

        def get_batch_tokens(self):
            self.calls += 1
            if self.calls == 1:
                return first_batch
            raise exc

    monkeypatch.setattr(slovorez_module, "FFTokenizer", FailingTokenizer)

    # model_batch=1: the first batch is inferred inside the loop, before the
    # failure; the results sit in the writer buffer until flush.
    with pytest.raises(type(exc)) as info:
        model.process_file("unused.txt", threaded_mode=False, model_batch=1)

    assert info.value is exc
    assert sorted(_written_words(model)) == sorted(first_batch)


@pytest.mark.parametrize("max_workers", [0, -1])
def test_threaded_mode_rejects_less_than_one_worker(model, small_text, max_workers):
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        model.process_file(small_text, threaded_mode=True, max_workers=max_workers)
