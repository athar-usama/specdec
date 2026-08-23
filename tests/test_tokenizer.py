import pytest

from specdec.tokenizer import BPETokenizer

SAMPLE = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps again and again and again."
)


def test_round_trip_encode_decode_is_lossless():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    ids = tok.encode(SAMPLE)
    assert tok.decode(ids) == SAMPLE


def test_training_reduces_token_count_below_raw_bytes():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    raw_byte_count = len(SAMPLE.encode("utf-8"))
    merged_count = len(tok.encode(SAMPLE))
    assert merged_count < raw_byte_count


def test_untrained_tokenizer_is_plain_bytes():
    tok = BPETokenizer()
    assert tok.vocab_size == 256
    ids = tok.encode("hi")
    assert ids == list(b"hi")


def test_unseen_text_still_round_trips_via_byte_fallback():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    weird = "éè unseen \U0001f600 text 123!"
    assert tok.decode(tok.encode(weird)) == weird


def test_save_and_load_round_trip(tmp_path):
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=280)
    path = tmp_path / "tok.pkl"
    tok.save(path)

    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(SAMPLE) == tok.encode(SAMPLE)


def test_vocab_size_below_256_is_rejected():
    tok = BPETokenizer()
    with pytest.raises(ValueError):
        tok.train(SAMPLE, vocab_size=100)
